import csv
import os
import tempfile
from pathlib import Path

from validate_submission import validate, EXPECTED_HEADER

EXPECTED = {"d0": 3, "d1": 4}


def _write(rows) -> str:
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return path


def test_valid_file_has_no_errors():
    path = _write([EXPECTED_HEADER, ["d0", "001"], ["d1", "0001"]])
    assert validate(path, EXPECTED) == []
    Path(path).unlink()


def test_wrong_header():
    path = _write([["Doc ID", "Pred"], ["d0", "001"]])
    errors = validate(path, EXPECTED)
    assert any("header" in e for e in errors)
    Path(path).unlink()


def test_duplicate_document_id():
    path = _write([EXPECTED_HEADER, ["d0", "001"], ["d0", "010"], ["d1", "0001"]])
    errors = validate(path, EXPECTED)
    assert any("duplicate" in e for e in errors)
    Path(path).unlink()


def test_missing_document_id():
    path = _write([EXPECTED_HEADER, ["d0", "001"]])
    errors = validate(path, EXPECTED)
    assert any("missing" in e for e in errors)
    Path(path).unlink()


def test_non_binary_prediction():
    path = _write([EXPECTED_HEADER, ["d0", "0x1"], ["d1", "0001"]])
    errors = validate(path, EXPECTED)
    assert any("not a binary string" in e for e in errors)
    Path(path).unlink()


def test_wrong_length():
    path = _write([EXPECTED_HEADER, ["d0", "01"], ["d1", "0001"]])
    errors = validate(path, EXPECTED)
    assert any("length" in e for e in errors)
    Path(path).unlink()


def test_missing_final_boundary():
    path = _write([EXPECTED_HEADER, ["d0", "010"], ["d1", "0001"]])
    errors = validate(path, EXPECTED)
    assert any("final character is not a boundary" in e for e in errors)
    Path(path).unlink()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
