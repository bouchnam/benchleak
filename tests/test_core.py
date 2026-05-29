import pytest

from benchleak.core import (
    ContaminationResult,
    compare_distributions,
    scan,
)


def test_compare_identical_distributions_is_chance():
    scores = [1.0, 2.0, 3.0, 4.0]
    auc, p = compare_distributions(scores, list(scores))
    assert auc == pytest.approx(0.5)
    assert p > 0.5


def test_compare_suspect_strictly_greater():
    # n=4 each: smallest one-sided p with full separation is 1/C(8,4) ~= 0.014.
    auc, p = compare_distributions([10.0, 11.0, 12.0, 13.0], [1.0, 2.0, 3.0, 4.0])
    assert auc == pytest.approx(1.0)
    assert p < 0.05


def test_compare_suspect_strictly_lower():
    auc, _ = compare_distributions([1.0, 2.0, 3.0], [10.0, 11.0, 12.0])
    assert auc == pytest.approx(0.0)


def test_compare_auc_matches_manual_count():
    # AUC = fraction of (suspect, reference) pairs where suspect > reference,
    # ties counting as 0.5. suspect=[2,4], reference=[1,3]:
    #   (2>1)=1, (2>3)=0, (4>1)=1, (4>3)=1  => 3/4 = 0.75
    auc, _ = compare_distributions([2.0, 4.0], [1.0, 3.0])
    assert auc == pytest.approx(0.75)


def test_compare_rejects_empty():
    with pytest.raises(ValueError):
        compare_distributions([], [1.0])
    with pytest.raises(ValueError):
        compare_distributions([1.0], [])


class _FakeDetector:
    """Replays a queue of canned score lists, one per score_batch call."""

    def __init__(self, *score_lists):
        self._queue = list(score_lists)

    def score_batch(self, texts):
        return self._queue.pop(0)


def test_scan_routes_scores_and_aggregates():
    detector = _FakeDetector([9.0, 8.0, 9.5, 8.5], [1.0, 2.0, 1.5, 2.5])
    result = scan(
        detector,
        benchmark_texts=["a", "b", "c", "d"],
        reference_texts=["w", "x", "y", "z"],
        detector_name="min-k",
        benchmark_name="gsm8k",
    )

    assert result.benchmark_scores == [9.0, 8.0, 9.5, 8.5]
    assert result.reference_scores == [1.0, 2.0, 1.5, 2.5]
    assert result.n_benchmark == 4
    assert result.n_reference == 4
    assert result.auc == pytest.approx(1.0)
    assert result.contaminated


def test_scan_clean_benchmark_not_flagged():
    overlap = [1.0, 2.0, 3.0, 4.0]
    detector = _FakeDetector(list(overlap), list(overlap))
    result = scan(
        detector,
        benchmark_texts=["a"] * 4,
        reference_texts=["b"] * 4,
        detector_name="min-k",
        benchmark_name="clean-bench",
    )

    assert result.auc == pytest.approx(0.5)
    assert not result.contaminated


def test_contaminated_requires_both_effect_and_significance():
    # High AUC but a non-significant p-value must not trip the flag.
    big_auc_weak_p = ContaminationResult(
        detector="min-k",
        benchmark="b",
        benchmark_scores=[1.0],
        reference_scores=[1.0],
        auc=0.9,
        p_value=0.2,
    )
    assert not big_auc_weak_p.contaminated

    # Significant but small effect must not trip it either.
    small_auc_strong_p = ContaminationResult(
        detector="min-k",
        benchmark="b",
        benchmark_scores=[1.0],
        reference_scores=[1.0],
        auc=0.52,
        p_value=0.001,
    )
    assert not small_auc_strong_p.contaminated
