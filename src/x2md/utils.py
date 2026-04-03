from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union


_PAGE_RE = re.compile(r"page_(\d+)\.", re.IGNORECASE)


def ensure_dir(path: Union[str, os.PathLike]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def page_sort_key(filename: str) -> tuple[int, str]:
    m = _PAGE_RE.search(filename)
    if m:
        return (int(m.group(1)), filename)
    return (10**9, filename)


def iter_files(
    directory: Union[str, os.PathLike],
    *,
    suffixes: Optional[Sequence[str]] = None,
) -> Iterator[Path]:
    d = Path(directory)
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(f"目录不存在或不是目录: {d}")

    for p in d.iterdir():
        if not p.is_file():
            continue
        if suffixes is None:
            yield p
            continue
        if p.suffix.lower() in {s.lower() for s in suffixes}:
            yield p


def sorted_files(
    directory: Union[str, os.PathLike],
    *,
    suffixes: Optional[Sequence[str]] = None,
    key=None,
) -> list[Path]:
    files = list(iter_files(directory, suffixes=suffixes))
    if key is None:
        files.sort(key=lambda p: page_sort_key(p.name))
    else:
        files.sort(key=key)
    return files


def strip_markdown_code_fences(text: str) -> str:
    text = text.replace("```markdown", "")
    return text.replace("```", "")


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 10
    initial_delay_s: float = 5.0
    backoff_factor: float = 2.0
