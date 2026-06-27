import math
import random

import pytest
import torch

from benchleak.detectors.perturb import T5MaskFillPerturber, WordSwapPerturber
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


# ---------------------------------------------------------------------------
# T5MaskFillPerturber — pure-logic tests (no model loaded)
# ---------------------------------------------------------------------------

def test_t5_protected_positions_digit_token():
    p = T5MaskFillPerturber()
    words = ["sold", "48", "clips"]
    assert p._protected_positions(words) == {1}


def test_t5_protected_positions_numeric_word():
    p = T5MaskFillPerturber()
    words = ["then", "half", "as", "many"]
    assert p._protected_positions(words) == {1}


def test_t5_protected_positions_currency_and_decimal():
    p = T5MaskFillPerturber()
    words = ["earns", "$12.50", "per", "twice,", "hour"]
    protected = p._protected_positions(words)
    assert 1 in protected  # $12.50 contains digits
    assert 3 in protected  # twice, after stripping punctuation


def test_t5_protected_positions_plain_words_unprotected():
    p = T5MaskFillPerturber()
    words = ["Natalia", "sold", "clips", "in", "April"]
    assert p._protected_positions(words) == set()


def test_t5_mask_never_moves_numeric_tokens_to_sentinels():
    # Run _mask with many seeds; protected words must never become a sentinel.
    math_text = "Natalia sold clips to 48 friends in April then half as many in May"
    words = math_text.split()
    p = T5MaskFillPerturber(mask_fraction=0.5, seed=0)
    protected = p._protected_positions(words)
    for seed in range(50):
        p._rng = random.Random(seed)
        masked, _ = p._mask(words)
        masked_words = masked.split()
        for w in masked_words:
            if w.startswith("<extra_id_"):
                # This sentinel replaced some original words — check none of those words
                # were protected by verifying protected words still appear literally.
                pass
        for i in protected:
            assert words[i] in masked_words, (
                f"protected word '{words[i]}' at index {i} was replaced by a sentinel"
            )


def test_t5_numbers_preserved_all_present():
    p = T5MaskFillPerturber()
    assert p._numbers_preserved("sold 48 clips in 2 months", "gave 48 items over 2 months") is True


def test_t5_numbers_preserved_number_missing():
    p = T5MaskFillPerturber()
    assert p._numbers_preserved("sold 48 clips", "gave some items") is False


def test_t5_numbers_preserved_no_numbers_in_original():
    p = T5MaskFillPerturber()
    assert p._numbers_preserved("no numbers here", "still no numbers there") is True


def test_t5_perturb_falls_back_when_number_changed(monkeypatch):
    # Patch _ensure_loaded + the model/tokenizer so no real T5 is downloaded.
    # The fake fill will return a text with "99" instead of "48".
    p = T5MaskFillPerturber(seed=0)
    monkeypatch.setattr(p, "_ensure_loaded", lambda: None)

    class _FakeTok:
        def __call__(self, text, return_tensors):
            return {"input_ids": torch.tensor([[0]])}

    class _FakeModel:
        device = None
        def generate(self, *a, **kw):
            return torch.tensor([[0]])

    class _FakeTokDecode:
        def decode(self, ids, skip_special_tokens):
            # Return something that _fills will parse as sentinel 0 → "bad fill"
            return "<extra_id_0> bad fill <extra_id_1>"

    p._tokenizer = type("T", (), {
        "__call__": lambda self, text, return_tensors: {"input_ids": torch.tensor([[0]])},
        "decode": lambda self, ids, skip_special_tokens: "<extra_id_0> bad fill <extra_id_1>",
    })()
    p._model = type("M", (), {
        "device": None,
        "generate": lambda self, ids, **kw: torch.tensor([[0]]),
    })()

    # _mask will produce one sentinel; _fills will return ["bad fill"];
    # _apply replaces the sentinel with "bad fill", dropping "48".
    # _numbers_preserved("... 48 ...", result_without_48) must be False → fallback to original.
    text = "sold 48 clips"
    results = p.perturb(text, n=3)
    # Every result must either preserve "48" or be the original text itself.
    for r in results:
        assert "48" in r, f"number '48' lost in paraphrase: {r!r}"
