import re

import pytest

from benchleak.data import BUNDLED_REFERENCES, load_reference


def test_bundled_reference_loads():
    texts = load_reference()
    assert len(texts) >= 20
    assert all(isinstance(t, str) and t for t in texts)
    assert all(not t.startswith("#") for t in texts)


def test_bundled_math_reference_loads():
    texts = load_reference(domain="math")
    assert len(texts) >= 20
    assert texts != load_reference()  # a different set, not the general one
    assert all("####" in t for t in texts)  # GSM8K-style final-answer marker


def test_bundled_math_reference_arithmetic_is_correct():
    """Every <<expression=value>> calculator annotation must actually hold.

    A wrong annotation would be a surprising token to any competent model and
    would bias the scan toward false positives.
    """
    for text in load_reference(domain="math"):
        annotations = re.findall(r"<<([^=<>]+)=([^<>]+)>>", text)
        assert annotations, f"passage without calculator annotations: {text[:60]}..."
        for expr, val in annotations:
            assert eval(expr) == pytest.approx(float(val)), f"<<{expr}={val}>> is wrong"


def test_unknown_domain_falls_back_to_general():
    assert load_reference(domain="no-such-domain") == load_reference()


def test_bundled_reference_files_exist_for_every_domain():
    for domain in BUNDLED_REFERENCES:
        assert load_reference(domain=domain)


def test_reference_limit():
    assert len(load_reference(limit=5)) == 5


def test_explicit_path_wins_over_domain(tmp_path):
    f = tmp_path / "ref.txt"
    f.write_text("custom passage\n", encoding="utf-8")
    assert load_reference(f, domain="math") == ["custom passage"]


def test_reference_from_custom_path(tmp_path):
    f = tmp_path / "ref.txt"
    f.write_text("# a comment\n\nfirst passage\nsecond passage\n", encoding="utf-8")
    assert load_reference(f) == ["first passage", "second passage"]


def test_reference_empty_file_errors(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("# only comments\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no reference texts"):
        load_reference(f)
