import pytest

from benchleak.loading import BENCHMARKS, BenchmarkSpec, extract_texts, resolve_spec


def test_resolve_known_benchmark_uses_registry_defaults():
    spec = resolve_spec("gsm8k")
    assert spec == BENCHMARKS["gsm8k"]
    assert spec.fields == ("question", "answer")


def test_resolve_known_benchmark_overrides():
    spec = resolve_spec("gsm8k", split="train", fields=["question"])
    assert spec.split == "train"
    assert spec.fields == ("question",)
    assert spec.config == "main"  # untouched default preserved


def test_resolve_unknown_benchmark_requires_fields():
    with pytest.raises(ValueError, match="unknown benchmark"):
        resolve_spec("some/private-dataset")


def test_resolve_unknown_benchmark_as_raw_path():
    spec = resolve_spec("some/private-dataset", fields=["text"], split="validation")
    assert spec == BenchmarkSpec("some/private-dataset", ("text",), config=None, split="validation")


def test_extract_texts_joins_fields():
    records = [{"question": "2+2?", "answer": "4"}, {"question": "3+3?", "answer": "6"}]
    assert extract_texts(records, ("question", "answer")) == ["2+2?\n4", "3+3?\n6"]


def test_extract_texts_skips_missing_and_empty_fields():
    records = [{"question": "q1", "answer": ""}, {"question": "q2"}]
    assert extract_texts(records, ("question", "answer")) == ["q1", "q2"]


def test_extract_texts_drops_fully_empty_records():
    records = [{"question": "", "answer": ""}, {"question": "kept", "answer": "x"}]
    assert extract_texts(records, ("question", "answer")) == ["kept\nx"]


def test_extract_texts_requires_fields():
    with pytest.raises(ValueError):
        extract_texts([{"a": "b"}], ())
