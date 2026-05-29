import math

import pytest
import torch

from benchleak.detectors.pretrain import (
    MinKProbDetector,
    min_k_prob,
    token_log_probs,
)


def test_token_log_probs_uniform_logits():
    # Flat logits over a 2-token vocab => every realised token has prob 0.5.
    logits = torch.zeros(3, 2)
    input_ids = torch.tensor([0, 1, 0])

    lps = token_log_probs(logits, input_ids)

    assert lps.shape == (2,)
    assert torch.allclose(lps, torch.full((2,), math.log(0.5)), atol=1e-6)


def test_token_log_probs_matches_manual_softmax():
    # logits[0] predicts input_ids[1] (== 1). p(1) = e^1 / (e^2 + e^1).
    logits = torch.tensor([[2.0, 1.0], [0.0, 0.0]])
    input_ids = torch.tensor([0, 1])

    expected = 1.0 - math.log(math.exp(2.0) + math.exp(1.0))

    lps = token_log_probs(logits, input_ids)
    assert lps.shape == (1,)
    assert lps.item() == pytest.approx(expected, abs=1e-6)


def test_token_log_probs_rejects_bad_shapes():
    with pytest.raises(ValueError):
        token_log_probs(torch.zeros(3), torch.tensor([0, 1, 2]))
    with pytest.raises(ValueError):
        token_log_probs(torch.zeros(3, 2), torch.zeros(3, 2))
    with pytest.raises(ValueError):
        token_log_probs(torch.zeros(4, 2), torch.tensor([0, 1, 0]))


def test_min_k_prob_keeps_lowest_tokens():
    # Bottom 20% of 10 tokens => 2 lowest log-probs: -5 and -4.
    lps = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    assert min_k_prob(lps, k=20.0) == pytest.approx((-5.0 + -4.0) / 2)


def test_min_k_prob_full_k_is_plain_mean():
    lps = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    assert min_k_prob(lps, k=100.0) == pytest.approx(lps.mean().item())


def test_min_k_prob_floors_token_count():
    # floor(10 * 25 / 100) == 2
    lps = torch.arange(0, -10, -1, dtype=torch.float32)  # 0, -1, ... -9
    assert min_k_prob(lps, k=25.0) == pytest.approx((-9.0 + -8.0) / 2)


def test_min_k_prob_keeps_at_least_one_token():
    # floor(3 * 1 / 100) == 0, but we always keep the single lowest token.
    lps = torch.tensor([-1.0, -7.0, -2.0])
    assert min_k_prob(lps, k=1.0) == pytest.approx(-7.0)


def test_min_k_prob_rejects_invalid_k():
    lps = torch.tensor([-1.0, -2.0])
    for bad in (0.0, -5.0, 150.0):
        with pytest.raises(ValueError):
            min_k_prob(lps, k=bad)


def test_min_k_prob_rejects_empty():
    with pytest.raises(ValueError):
        min_k_prob(torch.tensor([]))


def test_seen_text_scores_higher_than_unseen():
    # Core hypothesis: a memorised text has no low-probability outliers, so its
    # bottom-k%% mean log-prob is higher than that of an unseen text.
    seen = torch.tensor([-0.3, -0.2, -0.4, -0.5, -0.3])
    unseen = torch.tensor([-0.3, -0.2, -6.0, -0.5, -7.0])

    assert min_k_prob(seen, k=40.0) > min_k_prob(unseen, k=40.0)


class _FakeTokenizer:
    """Returns a fixed token sequence regardless of the input text."""

    def __init__(self, input_ids):
        self._input_ids = torch.tensor([input_ids])

    def __call__(self, text, **kwargs):
        return {"input_ids": self._input_ids}


class _FakeModel:
    """Returns preset logits, mimicking a transformers causal-LM forward pass."""

    device = "cpu"

    def __init__(self, logits):
        self._out = type("Output", (), {"logits": logits.unsqueeze(0)})()

    def __call__(self, input_ids):
        return self._out


def test_detector_score_matches_pure_functions():
    input_ids = [0, 1, 0, 1, 0]
    logits = torch.tensor(
        [
            [3.0, 0.5],
            [0.1, 2.0],
            [2.5, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    detector = MinKProbDetector(_FakeModel(logits), _FakeTokenizer(input_ids), k=50.0)

    expected = min_k_prob(token_log_probs(logits, torch.tensor(input_ids)), k=50.0)
    assert detector.score("anything") == pytest.approx(expected, abs=1e-6)


def test_detector_score_batch():
    detector = MinKProbDetector(
        _FakeModel(torch.zeros(3, 2)), _FakeTokenizer([0, 1, 0]), k=100.0
    )
    scores = detector.score_batch(["a", "b"])

    assert len(scores) == 2
    assert all(s == pytest.approx(math.log(0.5), abs=1e-6) for s in scores)
