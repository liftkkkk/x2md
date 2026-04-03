from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

from .utils import ensure_dir

logger = logging.getLogger(__name__)


def pdf_to_png(
    pdf_path: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    *,
    dpi: int = 300,
) -> list[Path]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

    out_dir = ensure_dir(output_dir)

    import fitz

    doc = fitz.open(str(pdf_path))
    try:
        total_pages = len(doc)
        logger.info("PDF总页数: %s", total_pages)
        scale = dpi / 72
        matrix = fitz.Matrix(scale, scale)

        outputs: list[Path] = []
        for page_index in range(total_pages):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=matrix)
            output_filename = f"page_{page_index + 1:03d}.png"
            output_path = out_dir / output_filename
            pix.save(str(output_path))
            outputs.append(output_path)
            logger.info("已保存第 %s/%s 页: %s", page_index + 1, total_pages, output_filename)
        return outputs
    finally:
        doc.close()


def pdf_to_text(
    pdf_path: Union[str, os.PathLike],
    output_path: Optional[Union[str, os.PathLike]] = None,
    *,
    page_separator: bool = True,
) -> Path:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

    if output_path is None:
        output_path = pdf_path.with_suffix(".txt")
    output_path = Path(output_path)
    if output_path.parent and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    import fitz

    doc = fitz.open(str(pdf_path))
    try:
        total_pages = len(doc)
        logger.info("PDF文件包含 %s 页", total_pages)

        chunks: list[str] = []
        for page_num in range(total_pages):
            page = doc[page_num]
            page_text = page.get_text()
            if page_separator and page_num > 0:
                chunks.append("\n" + "=" * 50 + f" 第{page_num + 1}页 " + "=" * 50 + "\n")
            chunks.append(page_text)

        output_path.write_text("".join(chunks), encoding="utf-8")
        logger.info("文本内容已保存到: %s", output_path)
        return output_path
    finally:
        doc.close()
