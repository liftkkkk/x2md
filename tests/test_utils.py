from x2md.utils import page_sort_key, strip_markdown_code_fences


def test_page_sort_key() -> None:
    assert page_sort_key("page_002.png") < page_sort_key("page_010.png")
    assert page_sort_key("page_002.png") < page_sort_key("zzz.png")


def test_strip_markdown_code_fences() -> None:
    assert strip_markdown_code_fences("```markdown\nx\n```") == "\nx\n"
