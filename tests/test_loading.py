import json

import pytest

from benchleak.loading import (
    BENCHMARKS,
    BenchmarkSpec,
    extract_texts,
    is_local_benchmark,
    load_local_benchmark,
    resolve_spec,
)


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


def test_is_local_benchmark(tmp_path):
    f = tmp_path / "bench.txt"
    f.write_text("x\n", encoding="utf-8")
    assert is_local_benchmark(str(f))
    assert not is_local_benchmark("gsm8k")
    assert not is_local_benchmark("namespace/dataset")


def test_local_txt_one_per_line(tmp_path):
    f = tmp_path / "bench.txt"
    f.write_text("first\n\nsecond\n", encoding="utf-8")
    assert load_local_benchmark(f) == ["first", "second"]


def test_local_jsonl_joins_fields(tmp_path):
    f = tmp_path / "bench.jsonl"
    f.write_text(
        json.dumps({"question": "q1", "answer": "a1"}) + "\n"
        + json.dumps({"question": "q2", "answer": "a2"}) + "\n",
        encoding="utf-8",
    )
    assert load_local_benchmark(f, fields=["question", "answer"]) == ["q1\na1", "q2\na2"]


def test_local_json_array(tmp_path):
    f = tmp_path / "bench.json"
    f.write_text(json.dumps([{"text": "one"}, {"text": "two"}]), encoding="utf-8")
    assert load_local_benchmark(f, fields=["text"]) == ["one", "two"]


def test_local_csv(tmp_path):
    f = tmp_path / "bench.csv"
    f.write_text("question,answer\nq1,a1\nq2,a2\n", encoding="utf-8")
    assert load_local_benchmark(f, fields=["question"]) == ["q1", "q2"]


def test_local_structured_requires_fields(tmp_path):
    f = tmp_path / "bench.jsonl"
    f.write_text(json.dumps({"text": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="--field is required"):
        load_local_benchmark(f)


def test_local_unsupported_suffix(tmp_path):
    f = tmp_path / "bench.parquet"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported benchmark file type"):
        load_local_benchmark(f, fields=["text"])


def test_local_limit_and_empty(tmp_path):
    f = tmp_path / "bench.txt"
    f.write_text("a\nb\nc\n", encoding="utf-8")
    assert load_local_benchmark(f, limit=2) == ["a", "b"]

    empty = tmp_path / "empty.txt"
    empty.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no benchmark texts"):
        load_local_benchmark(empty)
