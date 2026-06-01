import pytest

from benchleak.data import load_reference


def test_bundled_reference_loads():
    texts = load_reference()
    assert len(texts) >= 20
    assert all(isinstance(t, str) and t for t in texts)
    assert all(not t.startswith("#") for t in texts)


def test_reference_limit():
    assert len(load_reference(limit=5)) == 5


def test_reference_from_custom_path(tmp_path):
    f = tmp_path / "ref.txt"
    f.write_text("# a comment\n\nfirst passage\nsecond passage\n", encoding="utf-8")
    assert load_reference(f) == ["first passage", "second passage"]


def test_reference_empty_file_errors(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("# only comments\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no reference texts"):
        load_reference(f)
