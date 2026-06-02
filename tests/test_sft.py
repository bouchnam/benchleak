import math

import pytest
import torch

from benchleak.detectors.perturb import WordSwapPerturber
from benchleak.detectors.pretrain import token_log_probs
from benchleak.detectors.sft import (
    ProbVariationDetector,
    mean_log_likelihood,
    probabilistic_variation,
)


def test_mean_log_likelihood_matches_token_log_probs():
    logits = torch.tensor([[2.0, 1.0], [0.0, 0.5], [1.0, 0.0]])
    input_ids = torch.tensor([0, 1, 0])
    expected = token_log_probs(logits, input_ids).mean().item()
    assert mean_log_likelihood(logits, input_ids) == pytest.approx(expected)


def test_probabilistic_variation_is_original_minus_mean():
    assert probabilistic_variation(-1.0, [-2.0, -4.0]) == pytest.approx(-1.0 - (-3.0))


def test_probabilistic_variation_zero_when_neighbours_match():
    assert probabilistic_variation(-2.5, [-2.5, -2.5]) == pytest.approx(0.0)


def test_probabilistic_variation_rejects_empty():
    with pytest.raises(ValueError):
        probabilistic_variation(-1.0, [])


# Controllable fakes: each text maps to a single realised token whose log-prob is
# fixed by a known distribution, so the detector's score is hand-computable.
_PROBS = torch.tensor([0.4, 0.3, 0.2, 0.1])


class _FakeTokenizer:
    def __init__(self, mapping):
        self._mapping = mapping

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[0, self._mapping[text]]])}


class _FakeModel:
    """Returns fixed logits; only row 0 matters (predicts the second token)."""

    def __init__(self):
        row0 = torch.log(_PROBS)
        self._logits = torch.stack([row0, torch.zeros(4)])

    def __call__(self, input_ids):
        return type("Output", (), {"logits": self._logits.unsqueeze(0)})()


class _FakePerturber:
    def __init__(self, neighbours):
        self._neighbours = neighbours

    def perturb(self, text, n):
        return list(self._neighbours)


def test_detector_score_matches_hand_computation():
    mapping = {"orig": 0, "n1": 3, "n2": 2}  # token ids -> probs 0.4, 0.1, 0.2
    detector = ProbVariationDetector(
        _FakeModel(), _FakeTokenizer(mapping), _FakePerturber(["n1", "n2"]), n_perturbations=2
    )
    expected = math.log(0.4) - (math.log(0.1) + math.log(0.2)) / 2
    assert detector.score("orig") == pytest.approx(expected, abs=1e-6)


def test_detector_score_positive_when_original_peaks():
    # Original is the most probable token; neighbours are less probable -> peak.
    mapping = {"x": 0, "a": 2, "b": 3}
    detector = ProbVariationDetector(
        _FakeModel(), _FakeTokenizer(mapping), _FakePerturber(["a", "b"]), n_perturbations=2
    )
    assert detector.score("x") > 0


def test_detector_score_batch():
    mapping = {"x": 1, "a": 1, "b": 1}  # all identical -> variation 0
    detector = ProbVariationDetector(
        _FakeModel(), _FakeTokenizer(mapping), _FakePerturber(["a", "b"]), n_perturbations=2
    )
    scores = detector.score_batch(["x", "x"])
    assert len(scores) == 2
    assert all(s == pytest.approx(0.0, abs=1e-6) for s in scores)


def test_detector_rejects_bad_n_perturbations():
    with pytest.raises(ValueError):
        ProbVariationDetector(_FakeModel(), _FakeTokenizer({}), _FakePerturber([]), n_perturbations=0)


def test_word_swap_perturber_is_deterministic_and_in_vocab():
    text = "the quick brown fox jumps over the lazy dog"
    vocab = set(text.split())
    p1 = WordSwapPerturber(fraction=0.3, seed=7)
    p2 = WordSwapPerturber(fraction=0.3, seed=7)

    out1 = p1.perturb(text, 5)
    out2 = p2.perturb(text, 5)

    assert out1 == out2  # deterministic given seed
    assert len(out1) == 5
    for neighbour in out1:
        words = neighbour.split()
        assert len(words) == len(text.split())  # length preserved
        assert set(words) <= vocab  # only words from the original


def test_word_swap_perturber_short_text_returns_copies():
    assert WordSwapPerturber().perturb("solo", 3) == ["solo", "solo", "solo"]


def test_word_swap_perturber_rejects_bad_fraction():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            WordSwapPerturber(fraction=bad)
