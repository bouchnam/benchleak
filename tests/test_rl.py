import math

import pytest
import torch

from benchleak.detectors.rl import (
    SelfCritiqueDetector,
    penalized_cosine_similarity,
    step_entropy,
)


def test_step_entropy_uniform_is_log_vocab():
    # A uniform distribution over V tokens has entropy log(V).
    logits = torch.zeros(1, 4)
    assert step_entropy(logits)[0].item() == pytest.approx(math.log(4))


def test_step_entropy_one_hot_is_zero():
    # An almost-deterministic distribution has near-zero entropy.
    logits = torch.tensor([[100.0, 0.0, 0.0, 0.0]])
    assert step_entropy(logits)[0].item() == pytest.approx(0.0, abs=1e-6)


def test_step_entropy_one_row_per_step():
    logits = torch.zeros(3, 5)
    assert step_entropy(logits).shape == (3,)


def test_step_entropy_rejects_non_2d():
    with pytest.raises(ValueError):
        step_entropy(torch.zeros(4))


def test_penalized_cosine_identical_sequences_is_one():
    seq = [0.1, 0.5, 0.9]
    assert penalized_cosine_similarity(seq, seq) == pytest.approx(1.0)


def test_penalized_cosine_orthogonal_sequences_is_zero():
    assert penalized_cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_penalized_cosine_applies_length_penalty():
    # Same direction but different lengths: cosine is 1, scaled by min/max length.
    a = [1.0, 1.0, 1.0, 1.0]
    b = [1.0, 1.0]
    # padded b = [1, 1, 0, 0]; cos = 2 / (2 * sqrt(2)) = 1/sqrt(2); penalty = 2/4.
    expected = (1.0 / math.sqrt(2)) * (2 / 4)
    assert penalized_cosine_similarity(a, b) == pytest.approx(expected)


def test_penalized_cosine_empty_sequence_is_zero():
    assert penalized_cosine_similarity([], [1.0, 2.0]) == 0.0


def test_penalized_cosine_zero_norm_is_zero():
    assert penalized_cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


# A controllable fake model/tokenizer so the detector's plumbing (generate ->
# entropy -> similarity) is exercised without downloading a model. Each generate
# call returns a preset list of (1, vocab) score tensors.
class _FakeTokenizer:
    chat_template = None
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}

    def decode(self, ids, skip_special_tokens=True):
        return "response text"


class _FakeOutput:
    def __init__(self, scores, sequences):
        self.scores = scores
        self.sequences = sequences


class _FakeModel:
    device = None

    def __init__(self, scores_per_call):
        self._scores_per_call = list(scores_per_call)
        self._call = 0

    def generate(self, input_ids, **kwargs):
        scores = self._scores_per_call[self._call]
        self._call += 1
        new = torch.zeros((1, len(scores)), dtype=torch.long)
        sequences = torch.cat([input_ids, new], dim=1)
        return _FakeOutput(tuple(scores), sequences)


def _row(*values):
    return torch.tensor([list(values)])


def test_detector_score_one_when_both_responses_match():
    # Identical entropy curves for the initial and self-critique responses -> 1.0.
    call = [_row(0.0, 0.0), _row(10.0, 0.0)]  # entropies log(2), ~0
    detector = SelfCritiqueDetector(_FakeModel([call, call]), _FakeTokenizer())
    assert detector.score("a problem") == pytest.approx(1.0)


def test_detector_score_matches_pure_functions():
    call1 = [_row(0.0, 0.0), _row(0.0, 0.0)]      # two steps, entropy log(2) each
    call2 = [_row(0.0, 0.0)]                        # one step, entropy log(2)
    detector = SelfCritiqueDetector(_FakeModel([call1, call2]), _FakeTokenizer())
    e1 = step_entropy(torch.cat(call1)).tolist()
    e2 = step_entropy(torch.cat(call2)).tolist()
    assert detector.score("q") == pytest.approx(penalized_cosine_similarity(e1, e2))


def test_detector_rejects_bad_max_new_tokens():
    with pytest.raises(ValueError):
        SelfCritiqueDetector(_FakeModel([]), _FakeTokenizer(), max_new_tokens=0)


def test_detector_score_batch():
    call = [_row(0.0, 0.0)]
    model = _FakeModel([call, call, call, call])
    detector = SelfCritiqueDetector(model, _FakeTokenizer())
    scores = detector.score_batch(["q1", "q2"])
    assert len(scores) == 2
