"""End-to-end smoke test for the pre-training detector against a real model.

Runs the full benchleak pipeline (load model, load benchmark, load reference,
score both, report) on a small HuggingFace model and prints the per-sample scores
alongside the final verdict.

Default run uses CPU and a small sample for a fast first pass:

    python scripts/smoke.py

Override the defaults as needed:

    python scripts/smoke.py --model Qwen/Qwen2.5-0.5B --benchmark gsm8k --limit 25

Developer convenience script, not part of the installed package.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

from benchleak.core import scan
from benchleak.data import load_reference
from benchleak.detectors.pretrain import MinKProbDetector
from benchleak.loading import load_benchmark, load_model, resolve_spec
from benchleak.report import format_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--benchmark", default="gsm8k")
    p.add_argument("--reference", default=None, help="reference file; default = bundled set")
    p.add_argument("--limit", type=int, default=25, help="samples per side (default: 25, kept small for speed)")
    p.add_argument("--k", type=float, default=20.0)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--device", default=None, help="leave unset for CPU; try 'mps' to use the M1 GPU")
    p.add_argument("--show", type=int, default=8, help="how many per-sample scores to print per side")
    return p.parse_args()


def _summary(label: str, scores: list[float]) -> str:
    return (
        f"{label:>10}: n={len(scores):<3} "
        f"mean={statistics.mean(scores):+.3f} "
        f"median={statistics.median(scores):+.3f} "
        f"min={min(scores):+.3f} max={max(scores):+.3f}"
    )


def main() -> int:
    args = parse_args()
    spec = resolve_spec(args.benchmark)

    print(f"→ loading model {args.model} (device={args.device or 'cpu'}) ...", flush=True)
    t0 = time.perf_counter()
    model, tokenizer = load_model(args.model, device=args.device)
    print(f"  done in {time.perf_counter() - t0:.1f}s", flush=True)

    print(f"→ loading benchmark {spec.path} / {spec.split} (limit {args.limit}) ...", flush=True)
    benchmark_texts = load_benchmark(spec, limit=args.limit)
    reference_texts = load_reference(args.reference, limit=args.limit, domain=spec.domain)
    print(f"  {len(benchmark_texts)} benchmark, {len(reference_texts)} reference texts", flush=True)

    detector = MinKProbDetector(model, tokenizer, k=args.k, max_length=args.max_length)

    print("→ scoring (one forward pass per text) ...", flush=True)
    t0 = time.perf_counter()
    result = scan(
        detector,
        benchmark_texts,
        reference_texts,
        detector_name="min-k% prob",
        benchmark_name=args.benchmark,
    )
    elapsed = time.perf_counter() - t0
    n = result.n_benchmark + result.n_reference
    print(f"  scored {n} texts in {elapsed:.1f}s ({elapsed / n:.2f}s/text)\n", flush=True)

    # Per-sample scores for sanity-checking the raw signal, highest first.
    bench_sorted = sorted(result.benchmark_scores, reverse=True)
    ref_sorted = sorted(result.reference_scores, reverse=True)
    print(f"Top {args.show} Min-K%% scores (higher = looks more memorised):")
    print(f"  {'benchmark':>12} | {'reference':>12}")
    for i in range(min(args.show, len(bench_sorted), len(ref_sorted))):
        print(f"  {bench_sorted[i]:>12.3f} | {ref_sorted[i]:>12.3f}")
    print()
    print(_summary("benchmark", result.benchmark_scores))
    print(_summary("reference", result.reference_scores))
    print()

    print(format_report(result, model_id=args.model))
    if spec.domain == "general" and args.reference is None:
        print(
            "\nReminder: the bundled general reference is prose; against a "
            "narrow-domain benchmark a high AUC partly reflects domain, not only "
            "memorisation. Swap in a domain-matched --reference before trusting "
            "the verdict.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
