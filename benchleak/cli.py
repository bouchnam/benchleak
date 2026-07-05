"""Command line interface for benchleak.

    benchleak --model Qwen/Qwen2.5-0.5B --benchmark gsm8k

Loads a HuggingFace model and a benchmark, scores both the benchmark and a
reference set with the chosen detector (pre-training, fine-tuning, or RL phase),
and prints a contamination report. Exits non-zero when the benchmark is flagged
as likely contaminated.
"""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .core import scan
from .data import BUNDLED_REFERENCES, load_reference
from .detectors.perturb import T5MaskFillPerturber, WordSwapPerturber
from .detectors.pretrain import DEFAULT_K, MinKProbDetector
from .detectors.rl import DEFAULT_MAX_NEW_TOKENS, SelfCritiqueDetector
from .detectors.sft import DEFAULT_N_PERTURBATIONS, ProbVariationDetector
from .loading import is_local_benchmark, load_benchmark, load_local_benchmark, load_model, resolve_spec
from .report import format_report

DETECTOR_NAMES = {
    "pretrain": "min-k% prob",
    "sft": "prob-variation (spv-mia)",
    "rl": "self-critique",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchleak",
        description="Detect benchmark contamination in an LLM (pre-training, fine-tuning, or RL phase).",
    )
    parser.add_argument("--model", required=True, help="HuggingFace model id, e.g. Qwen/Qwen2.5-0.5B")
    parser.add_argument("--benchmark", required=True, help="benchmark name (gsm8k, math, ...), a Hub dataset path, or a local file (.txt/.jsonl/.json/.csv)")
    parser.add_argument("--detector", choices=("pretrain", "sft", "rl"), default="pretrain", help="which phase to test: pretrain (Min-K%% Prob), sft (probabilistic variation), or rl (self-critique). Default: pretrain")
    parser.add_argument("--reference", help="path to a reference-text file (one passage per line); defaults to the bundled set matching the benchmark's domain")
    parser.add_argument("--field", action="append", dest="fields", help="benchmark text column(s); repeatable. Required for an unknown benchmark")
    parser.add_argument("--config", help="dataset config/subset name")
    parser.add_argument("--split", help="dataset split (default depends on the benchmark)")
    parser.add_argument("--limit", type=int, default=200, help="max samples per side (default: 200)")
    parser.add_argument("--max-length", type=int, default=2048, help="truncate texts to this many tokens (default: 2048)")
    parser.add_argument("--device", help="device to place the model on, e.g. cuda or mps")
    parser.add_argument("--dtype", default="auto", help="model dtype passed to transformers (default: auto)")
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token for private/gated models; defaults to the HF_TOKEN env var or a cached huggingface-cli login",
    )

    pretrain = parser.add_argument_group("pretrain detector")
    pretrain.add_argument("--k", type=float, default=DEFAULT_K, help=f"Min-K%% percentage (default: {DEFAULT_K})")

    sft = parser.add_argument_group("sft detector")
    sft.add_argument("--perturber", choices=("t5", "word"), default="t5", help="paraphraser for probabilistic variation: t5 (downloads a T5 model, higher quality) or word (no extra model). Default: t5")
    sft.add_argument("--perturber-model", default="t5-base", help="T5 model id for the t5 perturber (default: t5-base)")
    sft.add_argument("--n-perturbations", type=int, default=DEFAULT_N_PERTURBATIONS, help=f"paraphrases per sample for the sft detector (default: {DEFAULT_N_PERTURBATIONS})")

    rl = parser.add_argument_group("rl detector")
    rl.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help=f"tokens to generate per response for the rl detector (default: {DEFAULT_MAX_NEW_TOKENS})")

    parser.add_argument("--version", action="version", version=f"benchleak {__version__}")
    return parser


def build_detector(args: argparse.Namespace, model, tokenizer):
    """Construct the detector selected by ``--detector``."""
    if args.detector == "pretrain":
        return MinKProbDetector(model, tokenizer, k=args.k, max_length=args.max_length)

    if args.detector == "rl":
        return SelfCritiqueDetector(
            model, tokenizer, max_new_tokens=args.max_new_tokens, max_length=args.max_length
        )

    if args.perturber == "t5":
        perturber = T5MaskFillPerturber(model_name=args.perturber_model, device=args.device)
    else:
        perturber = WordSwapPerturber()
    return ProbVariationDetector(
        model, tokenizer, perturber, n_perturbations=args.n_perturbations, max_length=args.max_length
    )


def run(args: argparse.Namespace) -> int:
    local = is_local_benchmark(args.benchmark)
    spec = None if local else resolve_spec(
        args.benchmark, config=args.config, split=args.split, fields=args.fields
    )

    print(f"Loading model {args.model} ...", file=sys.stderr)
    model, tokenizer = load_model(args.model, device=args.device, dtype=args.dtype, token=args.hf_token)

    if local:
        print(f"Loading local benchmark {args.benchmark} ...", file=sys.stderr)
        benchmark_texts = load_local_benchmark(args.benchmark, fields=args.fields, limit=args.limit)
    else:
        print(f"Loading benchmark {spec.path} ({spec.split}) ...", file=sys.stderr)
        benchmark_texts = load_benchmark(spec, limit=args.limit)

    domain = spec.domain if spec is not None else "general"
    if args.reference:
        reference_label = args.reference
    else:
        reference_label = f"bundled {BUNDLED_REFERENCES.get(domain, BUNDLED_REFERENCES['general'])}"
    reference_texts = load_reference(args.reference, limit=args.limit, domain=domain)
    print(f"Reference set: {reference_label}", file=sys.stderr)

    detector = build_detector(args, model, tokenizer)
    print(f"Scoring {len(benchmark_texts)} benchmark + {len(reference_texts)} reference samples ...", file=sys.stderr)
    result = scan(
        detector,
        benchmark_texts,
        reference_texts,
        detector_name=DETECTOR_NAMES[args.detector],
        benchmark_name=args.benchmark,
    )

    print(format_report(result, model_id=args.model, reference=reference_label))
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
