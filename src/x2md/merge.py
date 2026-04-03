from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Union

from .utils import ensure_dir, sorted_files, strip_markdown_code_fences

logger = logging.getLogger(__name__)


def merge_markdown_dir(
    input_dir: Union[str, os.PathLike],
    output_file: Union[str, os.PathLike],
    *,
    separator: str = "\n" + "=" * 50 + "\n",
    strip_fences: bool = True,
    include_headers: bool = False,
) -> Path:
    input_dir = Path(input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在或不是目录: {input_dir}")

    output_file = Path(output_file)
    if output_file.parent:
        ensure_dir(output_file.parent)

    files = sorted_files(input_dir, suffixes=[".md", ".markdown"])
    logger.info("找到 %s 个文件，开始合并...", len(files))

    with output_file.open("w", encoding="utf-8") as out_f:
        for i, p in enumerate(files, 1):
            content = p.read_text(encoding="utf-8")
            if strip_fences:
                content = strip_markdown_code_fences(content)
            if include_headers:
                out_f.write(f"--- {p.name} ---\n")
                out_f.write(separator)
            out_f.write(content)
            if not content.endswith("\n"):
                out_f.write("\n")
            if separator and not include_headers and i < len(files):
                out_f.write(separator)
            logger.info("已合并第 %s/%s 个文件: %s", i, len(files), p.name)

    logger.info("合并完成: %s", output_file)
    return output_file
