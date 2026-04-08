from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional, Union

from .utils import RetryPolicy, ensure_dir, sorted_files

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "qwen-vl-ocr-2025-11-20"
DEFAULT_PROMPT = "将双栏论文的内容转换为Markdown格式。"
DEFAULT_PROVIDER = "dashscope"


def _read_text_sidecar(image_path: Path) -> str:
    candidates = [
        image_path.with_suffix(".md"),
        image_path.with_suffix(".markdown"),
        image_path.with_suffix(".txt"),
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"file provider 需要同名 .md/.markdown/.txt 侧写文件: {', '.join(str(p) for p in candidates)}"
    )


def _dashscope_call(
    image_path: Path,
    *,
    api_key: str,
    model: str,
    prompt: str,
) -> str:
    import dashscope

    dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

    messages = [
        {
            "role": "user",
            "content": [{"image": str(image_path)}, {"text": prompt}],
        }
    ]
    response = dashscope.MultiModalConversation.call(
        api_key=api_key,
        model=model,
        messages=messages,
    )
    return response.output.choices[0].message.content[0]["text"]


def ocr_image_to_markdown_with_stats(
    image_path: Union[str, os.PathLike],
    *,
    provider: str = DEFAULT_PROVIDER,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
    retry: RetryPolicy = RetryPolicy(),
) -> tuple[str, int]:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    provider = provider.lower().strip()
    if provider == "file":
        return (_read_text_sidecar(image_path), 1)
    if provider != "dashscope":
        raise ValueError(f"不支持的 OCR provider: {provider}（支持：dashscope/file）")

    if api_key is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 DASHSCOPE_API_KEY，无法调用 OCR 接口。")

    delay = retry.initial_delay_s
    last_err: Optional[BaseException] = None
    for attempt in range(1, retry.max_retries + 1):
        try:
            return (_dashscope_call(image_path, api_key=api_key, model=model, prompt=prompt), attempt)
        except Exception as e:
            last_err = e
            if attempt >= retry.max_retries:
                break
            logger.warning("API调用失败 (尝试 %s/%s): %s", attempt, retry.max_retries, e)
            logger.info("%ss 后重试...", delay)
            time.sleep(delay)
            delay *= retry.backoff_factor

    raise RuntimeError(f"OCR 调用失败，已重试 {retry.max_retries} 次: {last_err}") from last_err


def ocr_image_to_markdown(
    image_path: Union[str, os.PathLike],
    *,
    provider: str = DEFAULT_PROVIDER,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
    retry: RetryPolicy = RetryPolicy(),
) -> str:
    text, _attempts = ocr_image_to_markdown_with_stats(
        image_path,
        provider=provider,
        api_key=api_key,
        model=model,
        prompt=prompt,
        retry=retry,
    )
    return text


def ocr_folder_to_markdown(
    input_dir: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    *,
    provider: str = DEFAULT_PROVIDER,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
    retry: RetryPolicy = RetryPolicy(),
    force: bool = False,
    manifest_path: Optional[Union[str, os.PathLike]] = None,
    keep_going: bool = False,
    fail_dir: Optional[Union[str, os.PathLike]] = None,
    cost_per_page: Optional[float] = None,
) -> list[Path]:
    input_dir = Path(input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在或不是目录: {input_dir}")

    out_dir = ensure_dir(output_dir)
    images = sorted_files(input_dir, suffixes=[".png"])
    logger.info("找到 %s 张图片，开始 OCR...", len(images))

    manifest_file: Optional[Path] = None
    if manifest_path is not None:
        manifest_file = Path(manifest_path)
        if manifest_file.parent:
            ensure_dir(manifest_file.parent)

    fail_out_dir: Optional[Path] = None
    if fail_dir is not None:
        fail_out_dir = ensure_dir(fail_dir)

    outputs: list[Path] = []
    mf = None
    if manifest_file is not None:
        mf = manifest_file.open("a", encoding="utf-8")
    try:
        for i, img in enumerate(images, 1):
            output_path = out_dir / (img.stem + ".md")
            if output_path.exists() and not force:
                outputs.append(output_path)
                logger.info("已存在结果，跳过第 %s/%s 张图片: %s", i, len(images), output_path.name)
                if mf is not None:
                    record = {
                        "status": "skipped",
                        "provider": provider,
                        "model": model,
                        "input_image": str(img),
                        "output_md": str(output_path),
                        "attempts": 0,
                        "duration_ms": 0,
                        "cost_estimated": 0.0,
                        "error": None,
                        "ts_ms": int(time.time() * 1000),
                    }
                    mf.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            logger.info("正在处理第 %s/%s 张图片: %s", i, len(images), img.name)
            started = time.perf_counter()
            attempts = 0
            err_str: Optional[str] = None
            md: Optional[str] = None
            try:
                md, attempts = ocr_image_to_markdown_with_stats(
                    img,
                    provider=provider,
                    api_key=api_key,
                    model=model,
                    prompt=prompt,
                    retry=retry,
                )
            except Exception as e:
                err_str = f"{type(e).__name__}: {e}"
                if fail_out_dir is not None:
                    ts_ms = int(time.time() * 1000)
                    dest_img = fail_out_dir / f"{img.stem}_{ts_ms}{img.suffix}"
                    shutil.copy2(img, dest_img)
                    meta = {
                        "provider": provider,
                        "model": model,
                        "prompt": prompt,
                        "input_image": str(img),
                        "saved_image": str(dest_img),
                        "error": err_str,
                        "ts_ms": ts_ms,
                    }
                    (fail_out_dir / f"{img.stem}_{ts_ms}.json").write_text(
                        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                    )
                if not keep_going:
                    raise
            finally:
                duration_ms = int((time.perf_counter() - started) * 1000)
                if mf is not None:
                    status = "ok" if err_str is None else "error"
                    cost_estimated = 0.0
                    if err_str is None and cost_per_page is not None and provider.lower().strip() == "dashscope":
                        cost_estimated = float(cost_per_page)
                    record = {
                        "status": status,
                        "provider": provider,
                        "model": model,
                        "input_image": str(img),
                        "output_md": str(output_path),
                        "attempts": attempts,
                        "duration_ms": duration_ms,
                        "cost_estimated": cost_estimated,
                        "error": err_str,
                        "ts_ms": int(time.time() * 1000),
                    }
                    mf.write(json.dumps(record, ensure_ascii=False) + "\n")

            if md is None:
                continue

            if not md.endswith("\n"):
                md += "\n"
            output_path.write_text(md, encoding="utf-8")
            outputs.append(output_path)
            logger.info("识别完成，结果保存到: %s", output_path.name)
    finally:
        if mf is not None:
            mf.close()

    logger.info("OCR 完成，结果目录: %s", out_dir)
    return outputs
