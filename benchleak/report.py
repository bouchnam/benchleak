"""Format a contamination result into a human-readable report.

Kept dependency-free and returning a plain string so it is trivial to test and to
pipe into a file; the CLI is free to add colour on top.
"""

from __future__ import annotations

from .core import ContaminationResult


def _severity(auc: float) -> str:
    """A coarse word for how far the benchmark separates from the reference."""
    if auc >= 0.85:
        return "VERY HIGH"
    if auc >= 0.70:
        return "HIGH"
    if auc >= 0.60:
        return "ELEVATED"
    return "LOW"


def format_report(result: ContaminationResult, *, model_id: str | None = None) -> str:
    """Render a single pre-training contamination result as plain text."""
    verdict = "LIKELY CONTAMINATED" if result.contaminated else "NO STRONG EVIDENCE"

    lines = [
        "benchleak: pre-training contamination report",
        "=" * 52,
        f"Model:       {model_id or 'n/a'}",
        f"Benchmark:   {result.benchmark}",
        f"Detector:    {result.detector}",
        f"Samples:     {result.n_benchmark} benchmark vs {result.n_reference} reference",
        "",
        f"Separation (AUC):   {result.auc:.3f}   [{_severity(result.auc)}]",
        f"Significance (p):   {result.p_value:.3g}",
        f"Flag thresholds:    AUC >= {result.auc_threshold}, p < {result.significance}",
        "",
        f"Verdict: {verdict}",
    ]

    if result.n_benchmark < 5 or result.n_reference < 5:
        lines.append(
            "Note: fewer than 5 samples per side. The significance test cannot "
            "reach p < 0.05. Use more samples for a trustworthy verdict.",
        )

    return "\n".join(lines)
