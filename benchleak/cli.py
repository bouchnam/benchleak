"""Command line interface for benchleak.

    benchleak --model Qwen/Qwen2.5-0.5B --benchmark gsm8k

Loads a HuggingFace model and a benchmark, scores both the benchmark and a
reference set with the Min-K% pre-training detector, and prints a contamination
report. Exits non-zero when the benchmark is flagged as likely contaminated.
"""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .core import scan
from .data import load_reference
from .detectors.pretrain import DEFAULT_K, MinKProbDetector
from .loading import load_benchmark, load_model, resolve_spec
from .report import format_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchleak",
        description="Detect pre-training benchmark contamination in an LLM (Min-K% Prob).",
    )
    parser.add_argument("--model", required=True, help="HuggingFace model id, e.g. Qwen/Qwen2.5-0.5B")
    parser.add_argument("--benchmark", required=True, help="benchmark name (gsm8k, math, ...) or a Hub dataset path")
    parser.add_argument("--reference", help="path to a reference-text file (one passage per line); defaults to the bundled set")
    parser.add_argument("--field", action="append", dest="fields", help="benchmark text column(s); repeatable. Required for an unknown benchmark")
    parser.add_argument("--config", help="dataset config/subset name")
    parser.add_argument("--split", help="dataset split (default depends on the benchmark)")
    parser.add_argument("--limit", type=int, default=200, help="max samples per side (default: 200)")
    parser.add_argument("--k", type=float, default=DEFAULT_K, help=f"Min-K%% percentage (default: {DEFAULT_K})")
    parser.add_argument("--max-length", type=int, default=2048, help="truncate texts to this many tokens (default: 2048)")
    parser.add_argument("--device", help="device to place the model on, e.g. cuda or mps")
    parser.add_argument("--dtype", default="auto", help="model dtype passed to transformers (default: auto)")
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token for private/gated models; defaults to the HF_TOKEN env var or a cached huggingface-cli login",
    )
    parser.add_argument("--version", action="version", version=f"benchleak {__version__}")
    return parser


def run(args: argparse.Namespace) -> int:
    spec = resolve_spec(args.benchmark, config=args.config, split=args.split, fields=args.fields)

    print(f"Loading model {args.model} ...", file=sys.stderr)
    model, tokenizer = load_model(args.model, device=args.device, dtype=args.dtype, token=args.hf_token)

    print(f"Loading benchmark {spec.path} ({spec.split}) ...", file=sys.stderr)
    benchmark_texts = load_benchmark(spec, limit=args.limit)
    reference_texts = load_reference(args.reference, limit=args.limit)

    detector = MinKProbDetector(model, tokenizer, k=args.k, max_length=args.max_length)
    print(f"Scoring {len(benchmark_texts)} benchmark + {len(reference_texts)} reference samples ...", file=sys.stderr)
    result = scan(
        detector,
        benchmark_texts,
        reference_texts,
        detector_name="min-k% prob",
        benchmark_name=args.benchmark,
    )

    print(format_report(result, model_id=args.model))
    return 1 if result.contaminated else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:  # surface a clean message instead of a traceback
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
