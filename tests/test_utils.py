import json
import threading
import time
import urllib.request
from pathlib import Path

from x2md.cli import build_http_server, main
from x2md.ocr import ocr_folder_to_markdown
from x2md.utils import page_sort_key, strip_markdown_code_fences


def test_page_sort_key() -> None:
    assert page_sort_key("page_002.png") < page_sort_key("page_010.png")
    assert page_sort_key("page_002.png") < page_sort_key("zzz.png")


def test_strip_markdown_code_fences() -> None:
    assert strip_markdown_code_fences("```markdown\nx\n```") == "x\n"
    assert strip_markdown_code_fences("a\n```python\nb\n```\n") == "a\n```python\nb\n```\n"


def test_ocr_file_provider_writes_manifest_and_skips(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()

    (in_dir / "page_001.png").write_bytes(b"")
    (in_dir / "page_001.md").write_text("hello", encoding="utf-8")

    manifest = tmp_path / "manifest.jsonl"
    ocr_folder_to_markdown(in_dir, out_dir, provider="file", manifest_path=manifest)

    assert (out_dir / "page_001.md").read_text(encoding="utf-8") == "hello\n"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["status"] == "ok"
    assert rec["attempts"] == 1

    ocr_folder_to_markdown(in_dir, out_dir, provider="file", manifest_path=manifest)
    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[1])
    assert rec["status"] == "skipped"
    assert rec["attempts"] == 0


def test_ocr_keep_going_writes_fail_artifacts_and_cost(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    fail_dir = tmp_path / "fail"
    in_dir.mkdir()

    (in_dir / "page_001.png").write_bytes(b"a")
    (in_dir / "page_001.md").write_text("ok", encoding="utf-8")
    (in_dir / "page_002.png").write_bytes(b"b")

    manifest = tmp_path / "manifest.jsonl"
    ocr_folder_to_markdown(
        in_dir,
        out_dir,
        provider="file",
        manifest_path=manifest,
        keep_going=True,
        fail_dir=fail_dir,
        cost_per_page=0.123,
    )

    assert (out_dir / "page_001.md").read_text(encoding="utf-8") == "ok\n"

    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rec1 = json.loads(lines[0])
    rec2 = json.loads(lines[1])
    assert rec1["status"] == "ok"
    assert rec1["cost_estimated"] == 0.0
    assert rec2["status"] == "error"

    saved_json = list(fail_dir.glob("page_002_*.json"))
    saved_png = list(fail_dir.glob("page_002_*.png"))
    assert len(saved_json) == 1
    assert len(saved_png) == 1


def test_report_summarizes_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"status": "ok", "duration_ms": 10, "attempts": 1, "cost_estimated": 0.2}),
                json.dumps({"status": "skipped", "duration_ms": 0, "attempts": 0, "cost_estimated": 0.0}),
                json.dumps({"status": "error", "duration_ms": 5, "attempts": 2, "cost_estimated": 0.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "report.json"
    code = main(["report", "--manifest", str(manifest), "-o", str(out)])
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["total"] == 3
    assert report["ok"] == 1
    assert report["skipped"] == 1
    assert report["error"] == 1
    assert report["cost_estimated_total"] == 0.2


def test_http_server_health_and_ocr_job(tmp_path: Path) -> None:
    server, _manager = build_http_server("127.0.0.1", 0, max_workers=2)
    host, port = server.server_address[0], server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health") as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            assert payload["status"] == "ok"

        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        (in_dir / "page_001.png").write_bytes(b"x")
        (in_dir / "page_001.md").write_text("hello", encoding="utf-8")

        req = urllib.request.Request(
            f"http://{host}:{port}/v1/ocr",
            method="POST",
            data=json.dumps(
                {
                    "input_dir": str(in_dir),
                    "output_dir": str(out_dir),
                    "provider": "file",
                },
                ensure_ascii=False,
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            created = json.loads(resp.read().decode("utf-8"))
            job_id = created["job_id"]

        deadline = time.time() + 3.0
        status = None
        while time.time() < deadline:
            with urllib.request.urlopen(f"http://{host}:{port}/v1/jobs/{job_id}") as resp:
                data = json.loads(resp.read().decode("utf-8"))
                status = data["status"]
                if status in {"finished", "error"}:
                    break
            time.sleep(0.05)

        assert status == "finished"
        assert (out_dir / "page_001.md").read_text(encoding="utf-8") == "hello\n"
    finally:
        server.shutdown()
        server.server_close()
