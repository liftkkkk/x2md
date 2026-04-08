from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from typing import Optional, Sequence

from .utils import RetryPolicy


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )


def _add_common_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="日志级别（DEBUG/INFO/WARNING/ERROR）",
    )


def _cmd_pdf2png(args: argparse.Namespace) -> int:
    from .pdf import pdf_to_png

    pdf_to_png(args.pdf_path, args.output_dir, dpi=args.dpi)
    return 0


def _cmd_pdf2txt(args: argparse.Namespace) -> int:
    from .pdf import pdf_to_text

    pdf_to_text(args.pdf_file, args.output, page_separator=not args.no_page_separator)
    return 0


def _cmd_ocr(args: argparse.Namespace) -> int:
    from .ocr import ocr_folder_to_markdown

    retry = RetryPolicy(
        max_retries=args.max_retries,
        initial_delay_s=args.initial_delay,
        backoff_factor=args.backoff,
    )
    ocr_folder_to_markdown(
        args.input_folder,
        args.output_folder,
        provider=args.provider,
        api_key=args.api_key,
        model=args.model,
        prompt=args.prompt,
        retry=retry,
        force=args.force,
        manifest_path=args.manifest,
        keep_going=args.keep_going,
        fail_dir=args.fail_dir,
        cost_per_page=args.cost_per_page,
    )
    return 0


def _cmd_merge(args: argparse.Namespace) -> int:
    from .merge import merge_markdown_dir

    merge_markdown_dir(
        args.input_directory,
        args.output_file_path,
        separator=args.separator,
        strip_fences=not args.keep_fences,
        include_headers=args.include_headers,
    )
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    from .batch import process_batch

    retry = RetryPolicy(
        max_retries=args.max_retries,
        initial_delay_s=args.initial_delay,
        backoff_factor=args.backoff,
    )
    process_batch(
        args.input_pdf_dir,
        output_base_dir=args.output_base_dir,
        limit=args.limit,
        dpi=args.dpi,
        provider=args.provider,
        model=args.model,
        prompt=args.prompt,
        api_key=args.api_key,
        retry=retry,
        force=args.force,
    )
    return 0


def _cmd_collect(args: argparse.Namespace) -> int:
    source_root = Path(args.source_root)
    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if not source_root.exists():
        raise FileNotFoundError(f"找不到目录: {source_root}")

    moved = 0
    for p in source_root.rglob("*.md"):
        if p.is_dir():
            continue
        if args.only_merged and not p.name.endswith("_merged.md"):
            continue
        dest = target_dir / p.name
        if dest.exists() and not args.overwrite:
            continue
        dest.write_bytes(p.read_bytes())
        if args.delete_source:
            p.unlink()
        moved += 1

    logging.getLogger(__name__).info("已收集 %s 个文件到: %s", moved, target_dir)
    return 0


def _cmd_video2audio(args: argparse.Namespace) -> int:
    from .video import video_to_wav

    video_to_wav(
        args.video_path,
        output_path=args.output,
        sample_rate=args.sample_rate,
        channels=args.channels,
    )
    return 0


def _cmd_download_model(args: argparse.Namespace) -> int:
    from .audio import download_asr_model

    download_asr_model(model_folder=args.model_folder)
    return 0


def _cmd_asr(args: argparse.Namespace) -> int:
    from .audio import transcribe_audio

    transcribe_audio(
        args.audio_path,
        output_json=args.output,
        model_folder=args.model_folder,
        model_path=args.model_path,
    )
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    import importlib.util
    import platform
    import shutil
    import sys

    log = logging.getLogger(__name__)

    def has_module(name: str) -> bool:
        return importlib.util.find_spec(name) is not None

    ok = True
    log.info("Python: %s (%s)", sys.version.split()[0], platform.platform())

    if has_module("fitz"):
        log.info("PyMuPDF: OK")
    else:
        log.error("PyMuPDF: 缺失（请安装 PyMuPDF>=1.23.0）")
        ok = False

    if has_module("dashscope"):
        log.info("dashscope: OK（OCR 可用）")
    else:
        log.warning("dashscope: 未安装（OCR 不可用，需 pip install -e '.[ocr]'）")

    if os.environ.get("DASHSCOPE_API_KEY"):
        log.info("DASHSCOPE_API_KEY: 已设置")
    else:
        log.warning("DASHSCOPE_API_KEY: 未设置（OCR 调用会失败）")

    if shutil.which("ffmpeg"):
        log.info("ffmpeg: OK（video2audio 可用）")
    else:
        log.warning("ffmpeg: 未找到（video2audio 不可用）")

    if has_module("modelscope"):
        log.info("modelscope: OK（asr 可用）")
    else:
        log.warning("modelscope: 未安装（asr 不可用，需 pip install -e '.[asr]'）")

    if ok:
        log.info("doctor: OK")
        return 0

    log.error("doctor: FAIL")
    return 1


def _cmd_eval(args: argparse.Namespace) -> int:
    import difflib
    import json

    log = logging.getLogger(__name__)
    pred_dir = Path(args.pred_dir)
    gold_dir = Path(args.gold_dir)

    if not pred_dir.exists() or not pred_dir.is_dir():
        raise FileNotFoundError(f"找不到目录: {pred_dir}")
    if not gold_dir.exists() or not gold_dir.is_dir():
        raise FileNotFoundError(f"找不到目录: {gold_dir}")

    pred_files = sorted([p for p in pred_dir.glob(args.glob) if p.is_file()])
    gold_files = {p.name for p in gold_dir.glob(args.glob) if p.is_file()}

    items: list[dict] = []
    matched = 0
    missing_gold = 0
    for p in pred_files:
        gold_path = gold_dir / p.name
        if not gold_path.exists():
            missing_gold += 1
            items.append({"file": p.name, "status": "missing_gold", "ratio": None})
            continue
        pred_text = p.read_text(encoding="utf-8", errors="replace")
        gold_text = gold_path.read_text(encoding="utf-8", errors="replace")
        ratio = difflib.SequenceMatcher(None, gold_text, pred_text).ratio()
        matched += 1
        items.append({"file": p.name, "status": "ok", "ratio": ratio})

    extra_gold = len(gold_files - {p.name for p in pred_files})
    avg_ratio = (
        sum(i["ratio"] for i in items if i["status"] == "ok" and i["ratio"] is not None) / matched
        if matched
        else 0.0
    )
    report = {
        "pred_dir": str(pred_dir),
        "gold_dir": str(gold_dir),
        "glob": args.glob,
        "files_pred": len(pred_files),
        "files_matched": matched,
        "files_missing_gold": missing_gold,
        "files_extra_gold": extra_gold,
        "avg_ratio": avg_ratio,
        "items": items,
    }

    if args.out is not None:
        out_path = Path(args.out)
        if out_path.parent:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log.info("评测报告已写入: %s", out_path)

    log.info(
        "评测完成：matched=%s missing_gold=%s extra_gold=%s avg_ratio=%.4f",
        matched,
        missing_gold,
        extra_gold,
        avg_ratio,
    )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    import statistics

    log = logging.getLogger(__name__)
    manifest_path = Path(args.manifest)
    if not manifest_path.exists() or not manifest_path.is_file():
        raise FileNotFoundError(f"找不到文件: {manifest_path}")

    total = 0
    ok = 0
    skipped = 0
    error = 0
    durations: list[int] = []
    costs: list[float] = []
    attempts_list: list[int] = []

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            status = rec.get("status")
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            elif status == "error":
                error += 1
            d = rec.get("duration_ms")
            if isinstance(d, int):
                durations.append(d)
            c = rec.get("cost_estimated")
            if isinstance(c, (int, float)):
                costs.append(float(c))
            a = rec.get("attempts")
            if isinstance(a, int):
                attempts_list.append(a)

    def pct(p: int, n: int) -> float:
        return (p / n) if n else 0.0

    summary = {
        "manifest": str(manifest_path),
        "total": total,
        "ok": ok,
        "skipped": skipped,
        "error": error,
        "ok_rate": pct(ok, total),
        "error_rate": pct(error, total),
        "duration_ms": {
            "avg": statistics.fmean(durations) if durations else 0.0,
            "p95": sorted(durations)[int(0.95 * (len(durations) - 1))] if durations else 0,
            "max": max(durations) if durations else 0,
        },
        "attempts": {
            "avg": statistics.fmean(attempts_list) if attempts_list else 0.0,
            "max": max(attempts_list) if attempts_list else 0,
        },
        "cost_estimated_total": sum(costs),
    }

    if args.out is not None:
        out_path = Path(args.out)
        if out_path.parent:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log.info("报告已写入: %s", out_path)
    else:
        log.info(json.dumps(summary, ensure_ascii=False))

    return 0


@dataclass
class Job:
    job_id: str
    kind: str
    status: str
    created_ts: float
    started_ts: Optional[float] = None
    finished_ts: Optional[float] = None
    error: Optional[str] = None
    result: Optional[dict[str, Any]] = None


class JobManager:
    def __init__(self, *, max_workers: int) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}

    def submit(self, kind: str, fn, payload: dict[str, Any]) -> Job:
        job = Job(job_id=uuid.uuid4().hex, kind=kind, status="queued", created_ts=time.time())
        with self._lock:
            self._jobs[job.job_id] = job

        def runner() -> None:
            started_ts = time.time()
            with self._lock:
                job.status = "running"
                job.started_ts = started_ts
            try:
                result = fn(payload)
                with self._lock:
                    job.status = "finished"
                    job.finished_ts = time.time()
                    job.result = result
            except Exception as e:
                with self._lock:
                    job.status = "error"
                    job.finished_ts = time.time()
                    job.error = f"{type(e).__name__}: {e}"

        self._executor.submit(runner)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())


def _job_ocr(payload: dict[str, Any]) -> dict[str, Any]:
    from .ocr import ocr_folder_to_markdown

    retry = RetryPolicy(
        max_retries=int(payload.get("max_retries", 10)),
        initial_delay_s=float(payload.get("initial_delay", 5.0)),
        backoff_factor=float(payload.get("backoff", 2.0)),
    )
    outputs = ocr_folder_to_markdown(
        payload["input_dir"],
        payload["output_dir"],
        provider=str(payload.get("provider", "dashscope")),
        api_key=payload.get("api_key"),
        model=str(payload.get("model", "qwen-vl-ocr-2025-11-20")),
        prompt=str(payload.get("prompt", "将双栏论文的内容转换为Markdown格式。")),
        retry=retry,
        force=bool(payload.get("force", False)),
        manifest_path=payload.get("manifest_path"),
        keep_going=bool(payload.get("keep_going", False)),
        fail_dir=payload.get("fail_dir"),
        cost_per_page=payload.get("cost_per_page"),
    )
    return {"outputs": [str(p) for p in outputs]}


def _job_batch(payload: dict[str, Any]) -> dict[str, Any]:
    from .batch import process_batch

    retry = RetryPolicy(
        max_retries=int(payload.get("max_retries", 10)),
        initial_delay_s=float(payload.get("initial_delay", 5.0)),
        backoff_factor=float(payload.get("backoff", 2.0)),
    )
    results = process_batch(
        payload["input_pdf_dir"],
        output_base_dir=payload["output_base_dir"],
        limit=payload.get("limit"),
        dpi=int(payload.get("dpi", 300)),
        provider=str(payload.get("provider", "dashscope")),
        model=str(payload.get("model", "qwen-vl-ocr-2025-11-20")),
        prompt=str(payload.get("prompt", "将双栏论文的内容转换为Markdown格式。")),
        api_key=payload.get("api_key"),
        retry=retry,
        force=bool(payload.get("force", False)),
    )
    return {"merged_files": [str(p) for p in results]}


def build_http_server(
    host: str,
    port: int,
    *,
    max_workers: int = 4,
) -> tuple[ThreadingHTTPServer, JobManager]:
    manager = JobManager(max_workers=max_workers)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            logging.getLogger(__name__).info("%s - %s", self.address_string(), format % args)

        def _send_json(self, status: int, obj: Any) -> None:
            raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b""
            if not body:
                return {}
            obj = json.loads(body.decode("utf-8"))
            if not isinstance(obj, dict):
                raise ValueError("JSON body 必须是 object")
            return obj

        def do_GET(self) -> None:
            p = urlparse(self.path)
            if p.path == "/health":
                self._send_json(200, {"status": "ok"})
                return

            if p.path == "/v1/jobs":
                jobs = []
                for job in manager.list():
                    jobs.append(
                        {
                            "job_id": job.job_id,
                            "kind": job.kind,
                            "status": job.status,
                            "created_ts": job.created_ts,
                            "started_ts": job.started_ts,
                            "finished_ts": job.finished_ts,
                            "error": job.error,
                        }
                    )
                self._send_json(200, {"jobs": jobs})
                return

            if p.path.startswith("/v1/jobs/"):
                job_id = p.path.split("/", 3)[-1]
                job_obj = manager.get(job_id)
                if job_obj is None:
                    self._send_json(404, {"error": "not_found"})
                    return
                self._send_json(
                    200,
                    {
                        "job_id": job_obj.job_id,
                        "kind": job_obj.kind,
                        "status": job_obj.status,
                        "created_ts": job_obj.created_ts,
                        "started_ts": job_obj.started_ts,
                        "finished_ts": job_obj.finished_ts,
                        "error": job_obj.error,
                        "result": job_obj.result,
                    },
                )
                return

            self._send_json(404, {"error": "not_found"})

        def do_POST(self) -> None:
            p = urlparse(self.path)
            try:
                payload = self._read_json()
            except Exception as e:
                self._send_json(400, {"error": f"bad_request: {type(e).__name__}: {e}"})
                return

            if p.path == "/v1/ocr":
                if "input_dir" not in payload or "output_dir" not in payload:
                    self._send_json(400, {"error": "missing_fields"})
                    return
                job = manager.submit("ocr", _job_ocr, payload)
                self._send_json(202, {"job_id": job.job_id})
                return

            if p.path == "/v1/batch":
                if "input_pdf_dir" not in payload or "output_base_dir" not in payload:
                    self._send_json(400, {"error": "missing_fields"})
                    return
                job = manager.submit("batch", _job_batch, payload)
                self._send_json(202, {"job_id": job.job_id})
                return

            self._send_json(404, {"error": "not_found"})

    server = ThreadingHTTPServer((host, port), Handler)
    return server, manager


def _cmd_serve(args: argparse.Namespace) -> int:
    log = logging.getLogger(__name__)
    server, _manager = build_http_server(args.host, args.port, max_workers=args.workers)
    host, port = server.server_address[0], server.server_address[1]
    log.info("listening on http://%s:%s", host, port)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="x2md", description="X to Markdown：PDF/图片/音视频 -> Markdown/TXT 工具集")
    _add_common_logging_args(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="运行环境自检（依赖/配置检查）")
    p.set_defaults(func=_cmd_doctor)

    p = sub.add_parser("eval", help="质量评测：对比 pred 与 gold（文件级相似度）")
    p.add_argument("--pred-dir", required=True, help="预测结果目录（.md）")
    p.add_argument("--gold-dir", required=True, help="标注/真值目录（.md）")
    p.add_argument("--glob", default="*.md", help="文件匹配模式（默认 *.md）")
    p.add_argument("-o", "--out", default=None, help="输出 JSON 报告路径（可选）")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("report", help="汇总 manifest.jsonl（成功率/耗时/成本）")
    p.add_argument("--manifest", required=True, help="manifest.jsonl 路径")
    p.add_argument("-o", "--out", default=None, help="输出 JSON 报告路径（可选）")
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("serve", help="启动 HTTP API 服务（本地摄取队列）")
    p.add_argument("--host", default="127.0.0.1", help="监听地址")
    p.add_argument("--port", type=int, default=8000, help="监听端口")
    p.add_argument("--workers", type=int, default=4, help="后台任务并发数")
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("pdf2png", help="PDF 转 PNG（高分辨率）")
    p.add_argument("pdf_path", help="输入 PDF 路径")
    p.add_argument("output_dir", help="输出图片目录")
    p.add_argument("--dpi", type=int, default=300, help="输出 DPI，默认 300")
    p.set_defaults(func=_cmd_pdf2png)

    p = sub.add_parser("ocr", help="PNG 批量 OCR -> Markdown")
    p.add_argument("input_folder", help="输入图片目录（PNG）")
    p.add_argument("output_folder", help="输出 Markdown 目录")
    p.add_argument("--provider", default="dashscope", choices=["dashscope", "file"], help="OCR 提供方")
    p.add_argument("--api-key", default=None, help="OCR API Key（默认读 DASHSCOPE_API_KEY）")
    p.add_argument("--model", default="qwen-vl-ocr-2025-11-20", help="OCR 模型名")
    p.add_argument("--prompt", default="将双栏论文的内容转换为Markdown格式。", help="OCR 提示词")
    p.add_argument("--max-retries", type=int, default=10, help="最大重试次数")
    p.add_argument("--initial-delay", type=float, default=5.0, help="初始重试延迟（秒）")
    p.add_argument("--backoff", type=float, default=2.0, help="指数退避倍率")
    p.add_argument("--force", action="store_true", help="覆盖已存在的输出（重跑）")
    p.add_argument("--manifest", default=None, help="写出 JSONL 运行清单（可选）")
    p.add_argument("--keep-going", action="store_true", help="遇到错误继续处理后续页面")
    p.add_argument("--fail-dir", default=None, help="失败样本落盘目录（可选）")
    p.add_argument("--cost-per-page", type=float, default=None, help="按页估算成本（仅 dashscope OK 页生效）")
    p.set_defaults(func=_cmd_ocr)

    p = sub.add_parser("merge", help="合并 Markdown 结果")
    p.add_argument("input_directory", help="输入目录（Markdown 文件）")
    p.add_argument("output_file_path", help="输出文件路径（.md）")
    p.add_argument(
        "--separator",
        default="\n" + "=" * 50 + "\n",
        help="文件内容之间分隔符",
    )
    p.add_argument("--keep-fences", action="store_true", help="不移除 ```/```markdown")
    p.add_argument("--include-headers", action="store_true", help="为每段内容添加来源文件名头")
    p.set_defaults(func=_cmd_merge)

    p = sub.add_parser("batch", help="批量处理论文：PDF -> PNG -> OCR -> 合并")
    p.add_argument(
        "--input-pdf-dir",
        default=str(Path(os.getcwd()) / "待处理论文"),
        help="输入 PDF 目录（默认 ./待处理论文）",
    )
    p.add_argument(
        "--output-base-dir",
        default=str(Path(os.getcwd()) / "处理结果"),
        help="输出根目录（默认 ./处理结果）",
    )
    p.add_argument("--limit", type=int, default=None, help="仅处理前 N 个文件（用于测试）")
    p.add_argument("--dpi", type=int, default=300, help="输出 DPI，默认 300")
    p.add_argument("--provider", default="dashscope", choices=["dashscope", "file"], help="OCR 提供方")
    p.add_argument("--api-key", default=None, help="OCR API Key（默认读 DASHSCOPE_API_KEY）")
    p.add_argument("--model", default="qwen-vl-ocr-2025-11-20", help="OCR 模型名")
    p.add_argument("--prompt", default="将双栏论文的内容转换为Markdown格式。", help="OCR 提示词")
    p.add_argument("--max-retries", type=int, default=10, help="最大重试次数")
    p.add_argument("--initial-delay", type=float, default=5.0, help="初始重试延迟（秒）")
    p.add_argument("--backoff", type=float, default=2.0, help="指数退避倍率")
    p.add_argument("--force", action="store_true", help="强制重跑所有步骤")
    p.set_defaults(func=_cmd_batch)

    p = sub.add_parser("collect", help="收集批处理产物到指定目录（便于整理/发布）")
    p.add_argument("--source-root", default=str(Path(os.getcwd()) / "处理结果"), help="来源根目录")
    p.add_argument("--target-dir", default=str(Path(os.getcwd())), help="目标目录")
    p.add_argument("--only-merged", action="store_true", help="仅收集 *_merged.md")
    p.add_argument("--overwrite", action="store_true", help="覆盖同名文件")
    p.add_argument("--delete-source", action="store_true", help="收集后删除源文件")
    p.set_defaults(func=_cmd_collect)

    p = sub.add_parser("video2audio", help="视频转 WAV（需要 ffmpeg）")
    p.add_argument("video_path", help="输入视频路径")
    p.add_argument("-o", "--output", default=None, help="输出 WAV 路径")
    p.add_argument("--sample-rate", type=int, default=44100, help="采样率，默认 44100")
    p.add_argument("--channels", type=int, default=2, help="声道数，默认 2")
    p.set_defaults(func=_cmd_video2audio)

    p = sub.add_parser("download-model", help="下载语音识别模型到本地目录")
    p.add_argument(
        "--model-folder",
        default=os.environ.get("X2MD_ASR_MODEL_FOLDER")
        or os.environ.get("X2MD_MODEL_FOLDER")
        or "model_folder",
        help="模型保存目录（默认 ./model_folder）",
    )
    p.set_defaults(func=_cmd_download_model)

    p = sub.add_parser("asr", help="本地语音识别（需要 modelscope，本地已下载模型）")
    p.add_argument("audio_path", help="输入音频路径")
    p.add_argument("-o", "--output", default="recognition_result.json", help="输出 JSON 路径")
    p.add_argument(
        "--model-folder",
        default=os.environ.get("X2MD_ASR_MODEL_FOLDER")
        or os.environ.get("X2MD_MODEL_FOLDER")
        or "model_folder",
        help="模型目录（默认 ./model_folder）",
    )
    p.add_argument(
        "--model-path",
        default=os.environ.get("X2MD_ASR_MODEL_PATH"),
        help="直接指定模型目录（覆盖 --model-folder）",
    )
    p.set_defaults(func=_cmd_asr)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    return int(args.func(args))
