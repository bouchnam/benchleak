"""Loading models and benchmarks from the HuggingFace Hub.

These are the only parts of benchleak that touch the network and the heavy
``transformers``/``datasets`` stack, so the imports are deferred to call time:
importing benchleak, and running its unit tests, stays cheap. The pure
record-to-text logic is split out into :func:`extract_texts` so it can be tested
without downloading anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class BenchmarkSpec:
    """Where a known benchmark lives and which fields carry its text."""

    path: str
    fields: tuple[str, ...]
    config: str | None = None
    split: str = "test"


# Benchmarks the registry resolves without the caller spelling out every field.
BENCHMARKS: dict[str, BenchmarkSpec] = {
    "gsm8k": BenchmarkSpec("openai/gsm8k", ("question", "answer"), config="main", split="test"),
    "math": BenchmarkSpec("hendrycks/competition_math", ("problem", "solution"), split="test"),
    "arc-challenge": BenchmarkSpec(
        "allenai/ai2_arc", ("question",), config="ARC-Challenge", split="test"
    ),
    "truthfulqa": BenchmarkSpec(
        "truthfulqa/truthful_qa", ("question", "best_answer"), config="generation", split="validation"
    ),
}


def resolve_spec(
    benchmark: str,
    *,
    config: str | None = None,
    split: str | None = None,
    fields: Sequence[str] | None = None,
) -> BenchmarkSpec:
    """Look ``benchmark`` up in the registry, or treat it as a raw Hub path.

    For an unknown benchmark the caller must supply ``fields`` (which columns hold
    the text); known benchmarks fall back to their registered defaults, and any of
    ``config``/``split``/``fields`` passed in override those defaults.
    """
    base = BENCHMARKS.get(benchmark)
    if base is None:
        if not fields:
            known = ", ".join(sorted(BENCHMARKS))
            raise ValueError(
                f"unknown benchmark {benchmark!r}; pass --field to name its text "
                f"column(s), or use one of: {known}"
            )
        return BenchmarkSpec(
            path=benchmark,
            fields=tuple(fields),
            config=config,
            split=split or "test",
        )

    return BenchmarkSpec(
        path=base.path,
        fields=tuple(fields) if fields else base.fields,
        config=config if config is not None else base.config,
        split=split if split is not None else base.split,
    )


def extract_texts(records: Iterable[dict], fields: Sequence[str], *, separator: str = "\n") -> list[str]:
    """Join the named ``fields`` of each record into one text per record.

    Missing or empty fields are skipped; a record contributing no text at all is
    dropped rather than emitted as an empty string.
    """
    if not fields:
        raise ValueError("need at least one field to extract benchmark text")

    texts = []
    for record in records:
        parts = [str(record[f]).strip() for f in fields if record.get(f) not in (None, "")]
        joined = separator.join(p for p in parts if p)
        if joined:
            texts.append(joined)
    return texts


def _import_load_dataset():
    """Import ``datasets.load_dataset``, with a clear message if lzma is missing.

    Some Python builds (notably pyenv on macOS without the xz library present at
    build time) ship without the ``_lzma`` stdlib module, which ``datasets``
    imports on load. The raw error is cryptic, so it is translated here.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # ModuleNotFoundError is a subclass
        if "lzma" in str(exc):
            raise RuntimeError(
                "Loading benchmarks requires Python's lzma module, which is "
                "missing from this interpreter (common with pyenv builds on "
                "macOS that were compiled without the xz library). Fix it by "
                "installing xz and rebuilding Python, e.g.:\n"
                '  brew install xz\n'
                '  LDFLAGS="-L$(brew --prefix xz)/lib" '
                'CPPFLAGS="-I$(brew --prefix xz)/include" '
                "pyenv install -f <your-python-version>\n"
                "then reuse or recreate your virtual environment."
            ) from exc
        raise
    return load_dataset


def load_benchmark(spec: BenchmarkSpec, *, limit: int | None = None) -> list[str]:
    """Download a benchmark split and return one text string per example."""
    load_dataset = _import_load_dataset()  # deferred: heavy, network-bound

    dataset = load_dataset(spec.path, spec.config, split=spec.split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return extract_texts(dataset, spec.fields)


def load_model(model_id: str, *, device: str | None = None, dtype: str = "auto", token: str | None = None):
    """Load a causal-LM and its tokenizer from the Hub, ready for scoring.

    ``token`` is a HuggingFace access token for private or gated repositories.
    When omitted, transformers falls back to the ``HF_TOKEN`` environment variable
    or a cached ``huggingface-cli login`` credential.
    """
    import torch  # noqa: F401  (ensures a clear error if torch is missing)
    from transformers import AutoModelForCausalLM, AutoTokenizer  # deferred

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, token=token)
    model.eval()
    if device is not None:
        model.to(device)
    return model, tokenizer
