from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
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
        api_key=args.api_key,
        model=args.model,
        prompt=args.prompt,
        retry=retry,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="x2md", description="X to Markdown：PDF/图片/音视频 -> Markdown/TXT 工具集")
    _add_common_logging_args(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pdf2png", help="PDF 转 PNG（高分辨率）")
    p.add_argument("pdf_path", help="输入 PDF 路径")
    p.add_argument("output_dir", help="输出图片目录")
    p.add_argument("--dpi", type=int, default=300, help="输出 DPI，默认 300")
    p.set_defaults(func=_cmd_pdf2png)

    p = sub.add_parser("ocr", help="PNG 批量 OCR -> Markdown")
    p.add_argument("input_folder", help="输入图片目录（PNG）")
    p.add_argument("output_folder", help="输出 Markdown 目录")
    p.add_argument("--api-key", default=None, help="DashScope API Key（默认读 DASHSCOPE_API_KEY）")
    p.add_argument("--model", default="qwen-vl-ocr-2025-11-20", help="OCR 模型名")
    p.add_argument("--prompt", default="将双栏论文的内容转换为Markdown格式。", help="OCR 提示词")
    p.add_argument("--max-retries", type=int, default=10, help="最大重试次数")
    p.add_argument("--initial-delay", type=float, default=5.0, help="初始重试延迟（秒）")
    p.add_argument("--backoff", type=float, default=2.0, help="指数退避倍率")
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
    p.add_argument("--api-key", default=None, help="DashScope API Key（默认读 DASHSCOPE_API_KEY）")
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
