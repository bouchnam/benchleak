"""Orchestration for a reference-based contamination scan.

A single Min-K% score is meaningless in isolation: its scale depends on the
model, tokenizer and text length. We make it interpretable by scoring a second,
*reference* set of texts known to post-date the model's training, then asking how
strongly the benchmark's scores separate from the reference's.

That separation is the AUC of a membership classifier built from the scores,
which the Mann-Whitney U statistic yields directly (AUC = U / n_suspect /
n_reference) alongside a significance test. AUC ~= 0.5 means the benchmark looks
just like fresh data (no signal); AUC well above 0.5 means benchmark samples
score systematically higher, the memorisation signature of contamination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from scipy.stats import mannwhitneyu

# Verdict heuristics: flag only when the separation is both sizeable and unlikely
# to be noise. These are deliberately conservative defaults, not hard science.
DEFAULT_AUC_THRESHOLD = 0.6
DEFAULT_SIGNIFICANCE = 0.05


class Detector(Protocol):
    """Anything that turns texts into per-sample contamination scores."""

    def score_batch(self, texts: Sequence[str]) -> list[float]: ...


@dataclass
class ContaminationResult:
    detector: str
    benchmark: str
    benchmark_scores: list[float]
    reference_scores: list[float]
    auc: float
    p_value: float
    auc_threshold: float = DEFAULT_AUC_THRESHOLD
    significance: float = DEFAULT_SIGNIFICANCE

    @property
    def n_benchmark(self) -> int:
        return len(self.benchmark_scores)

    @property
    def n_reference(self) -> int:
        return len(self.reference_scores)

    @property
    def contaminated(self) -> bool:
        """Benchmark scores separate from reference both clearly and significantly."""
        return self.auc >= self.auc_threshold and self.p_value < self.significance


def compare_distributions(
    suspect: Sequence[float], reference: Sequence[float]
) -> tuple[float, float]:
    """Test whether ``suspect`` scores stochastically exceed ``reference`` scores.

    Returns ``(auc, p_value)`` where ``auc`` is the probability that a randomly
    drawn suspect score is higher than a randomly drawn reference score — 0.5 under
    the null, approaching 1.0 as the benchmark looks more memorised.
    """
    if not suspect or not reference:
        raise ValueError("both suspect and reference score sets must be non-empty")

    result = mannwhitneyu(suspect, reference, alternative="greater")
    auc = result.statistic / (len(suspect) * len(reference))
    return auc, result.pvalue


def scan(
    detector: Detector,
    benchmark_texts: Sequence[str],
    reference_texts: Sequence[str],
    *,
    detector_name: str,
    benchmark_name: str,
    auc_threshold: float = DEFAULT_AUC_THRESHOLD,
    significance: float = DEFAULT_SIGNIFICANCE,
) -> ContaminationResult:
    """Score a benchmark against a reference set and quantify the separation."""
    benchmark_scores = detector.score_batch(benchmark_texts)
    reference_scores = detector.score_batch(reference_texts)
    auc, p_value = compare_distributions(benchmark_scores, reference_scores)

    return ContaminationResult(
        detector=detector_name,
        benchmark=benchmark_name,
        benchmark_scores=benchmark_scores,
        reference_scores=reference_scores,
        auc=auc,
        p_value=p_value,
        auc_threshold=auc_threshold,
        significance=significance,
    )
