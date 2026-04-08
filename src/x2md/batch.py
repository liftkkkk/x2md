from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

from .merge import merge_markdown_dir
from .ocr import DEFAULT_MODEL, DEFAULT_PROMPT, DEFAULT_PROVIDER, ocr_folder_to_markdown
from .pdf import pdf_to_png
from .utils import RetryPolicy, ensure_dir, page_sort_key

logger = logging.getLogger(__name__)


def process_single_pdf(
    pdf_path: Union[str, os.PathLike],
    *,
    output_base_dir: Union[str, os.PathLike],
    dpi: int = 300,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
    api_key: Optional[str] = None,
    retry: RetryPolicy = RetryPolicy(),
    force: bool = False,
    keep_going: bool = True,
    fail_dir: Optional[Union[str, os.PathLike]] = None,
    cost_per_page: Optional[float] = None,
) -> Path:
    pdf_path = Path(pdf_path)
    pdf_stem = pdf_path.stem

    paper_output_dir = ensure_dir(Path(output_base_dir) / pdf_stem)
    images_dir = ensure_dir(paper_output_dir / "images")
    markdown_dir = ensure_dir(paper_output_dir / "markdown")
    merged_file = paper_output_dir / f"{pdf_stem}_merged.md"

    logger.info("开始处理: %s", pdf_path.name)

    if merged_file.exists() and not force:
        logger.info("已存在合并结果，跳过: %s", merged_file)
        return merged_file

    if force or not any(images_dir.glob("*.png")):
        pdf_to_png(pdf_path, images_dir, dpi=dpi)
    else:
        logger.info("images 已存在，跳过 PDF 转图片: %s", images_dir)

    ocr_folder_to_markdown(
        images_dir,
        markdown_dir,
        provider=provider,
        api_key=api_key,
        model=model,
        prompt=prompt,
        retry=retry,
        force=force,
        keep_going=keep_going,
        fail_dir=fail_dir,
        manifest_path=str(paper_output_dir / "ocr_manifest.jsonl"),
        cost_per_page=cost_per_page,
    )

    merge_markdown_dir(markdown_dir, merged_file, strip_fences=True, include_headers=False)
    logger.info("处理完成: %s", merged_file)
    return merged_file


def process_batch(
    input_pdf_dir: Union[str, os.PathLike],
    *,
    output_base_dir: Union[str, os.PathLike],
    limit: Optional[int] = None,
    dpi: int = 300,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
    api_key: Optional[str] = None,
    retry: RetryPolicy = RetryPolicy(),
    force: bool = False,
) -> list[Path]:
    input_pdf_dir = Path(input_pdf_dir)
    if not input_pdf_dir.exists() or not input_pdf_dir.is_dir():
        raise FileNotFoundError(f"找不到输入目录: {input_pdf_dir}")

    ensure_dir(output_base_dir)

    pdf_files = sorted(
        [p for p in input_pdf_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"],
        key=lambda p: page_sort_key(p.name),
    )
    if limit is not None:
        pdf_files = pdf_files[:limit]

    if not pdf_files:
        logger.info("目录中没有找到 PDF: %s", input_pdf_dir)
        return []

    logger.info("找到 %s 个 PDF，开始批处理...", len(pdf_files))
    results: list[Path] = []
    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info("进度: %s/%s", i, len(pdf_files))
        try:
            merged = process_single_pdf(
                pdf_path,
                output_base_dir=output_base_dir,
                dpi=dpi,
                provider=provider,
                model=model,
                prompt=prompt,
                api_key=api_key,
                retry=retry,
                force=force,
            )
            results.append(merged)
        except Exception:
            logger.exception("处理失败，跳过: %s", pdf_path.name)

    logger.info("批处理完成，结果目录: %s", output_base_dir)
    return results
