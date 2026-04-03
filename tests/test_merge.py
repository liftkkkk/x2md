from pathlib import Path

from x2md.merge import merge_markdown_dir


def test_merge_strips_code_fences(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    in_dir.mkdir()

    (in_dir / "page_001.md").write_text("```markdown\nhello\n```\n", encoding="utf-8")
    (in_dir / "page_002.md").write_text("world\n", encoding="utf-8")

    out = tmp_path / "out.md"
    merge_markdown_dir(in_dir, out, separator="\n---\n", strip_fences=True, include_headers=False)

    merged = out.read_text(encoding="utf-8")
    assert "```" not in merged
    assert "hello" in merged
    assert "world" in merged
