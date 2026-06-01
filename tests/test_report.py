from benchleak.core import ContaminationResult
from benchleak.report import format_report


def _result(auc, p, n=10):
    return ContaminationResult(
        detector="min-k% prob",
        benchmark="gsm8k",
        benchmark_scores=[0.0] * n,
        reference_scores=[0.0] * n,
        auc=auc,
        p_value=p,
    )


def test_report_flags_contaminated():
    text = format_report(_result(0.8, 0.001), model_id="acme/model")
    assert "LIKELY CONTAMINATED" in text
    assert "acme/model" in text
    assert "gsm8k" in text
    assert "HIGH" in text


def test_report_clean_verdict():
    text = format_report(_result(0.52, 0.4))
    assert "NO STRONG EVIDENCE" in text
    assert "n/a" in text  # model_id omitted


def test_report_warns_on_small_samples():
    text = format_report(_result(0.9, 0.04, n=3))
    assert "fewer than 5 samples" in text
