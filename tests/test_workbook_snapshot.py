"""Keep the reader workbook tied to the canonical audit tables."""
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1] / "reproducibility"


def test_workbook_snapshot_matches_manifest_and_source_tables():
    manifest = json.loads((ROOT / "workbook_manifest.json").read_text(encoding="utf-8"))
    workbook = ROOT / manifest["workbook"]
    assert hashlib.sha256(workbook.read_bytes()).hexdigest() == manifest["sha256"]
    with zipfile.ZipFile(workbook) as archive:
        assert archive.testzip() is None
    for sheet in manifest["sheets"]:
        text = (ROOT / sheet["source"]).read_text(encoding="utf-8-sig")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == sheet["sha256_normalized"]
        rows = list(csv.reader(io.StringIO(text)))
        assert len(rows) - 1 == sheet["rows"]
        assert len(rows[0]) == sheet["columns"]
