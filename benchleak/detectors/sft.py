"""Probabilistic-variation SFT contamination detector (Fu et al. 2024).

Adapts the membership signal of SPV-MIA ("Membership Inference Attacks against
Fine-tuned Large Language Models via Self-prompt Calibration", NeurIPS 2024,
arXiv:2311.06062). The paper's insight: a probability *value* is an unreliable
membership signal because it only separates members from non-members when the
model is overfit. A more robust signal is whether a text sits on a local *maximum*
of the model's probability landscape, which is the signature of memorisation and
survives even without overfitting.

For a text x, that curvature is approximated by paraphrasing x into nearby
"neighbour" texts and comparing log-likelihoods (the practical neighbour form of
Eq. 10 in the paper):

    pv(x) = loglik(x) - mean_n loglik(paraphrase_n)

A text the model was fine-tuned on is a local peak, so its log-likelihood exceeds
that of its paraphrases and pv(x) is large and positive. An unseen text sits on a
flat slope, so pv(x) is near zero. Higher score therefore means more likely
memorised, matching the convention of the Min-K% detector, so the same
reference-based ``scan`` aggregates it.

Two deliberate departures from the paper, both to keep the tool light enough to
run locally in minutes (see docs/how-it-works-sft.md):
  - The self-prompt *reference model* (which the paper fine-tunes to calibrate) is
    replaced by benchleak's reference *dataset* comparison in ``core.scan``.
  - The symmetric second-derivative form is reduced to the asymmetric neighbour
    form the paper itself shows it rediscovers.
"""

from __future__ import annotations

from statistics import fmean
from typing import Protocol, Sequence

import torch

from .pretrain import token_log_probs

DEFAULT_N_PERTURBATIONS = 10


def mean_log_likelihood(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    """Mean per-token log-probability the model assigns to ``input_ids``.

    Length-normalised so texts of different lengths (an original and its
    paraphrases) are comparable. Built on the same next-token log-probabilities
    used by the Min-K% detector.
    """
    return token_log_probs(logits, input_ids).mean().item()


def probabilistic_variation(original_ll: float, perturbed_lls: Sequence[float]) -> float:
    """Curvature signal: how much the original log-likelihood peaks above its neighbours.

    ``original_ll`` is the log-likelihood of the text; ``perturbed_lls`` are the
    log-likelihoods of its paraphrases. A positive result means the text is a local
    maximum (the memorisation signature); near zero means it is not.
    """
    if not perturbed_lls:
        raise ValueError("need at least one perturbed log-likelihood to measure variation")
    return original_ll - fmean(perturbed_lls)


class Perturber(Protocol):
    """Generates paraphrases ("neighbours") of a text in the probability landscape."""

    def perturb(self, text: str, n: int) -> list[str]: ...


class ProbVariationDetector:
    """Score texts for fine-tuning membership via probabilistic variation.

    Like the Min-K% detector, this is agnostic to the model's origin: any object
    exposing a ``transformers``-style ``model(input_ids).logits`` forward pass and a
    matching tokenizer works, and the ``perturber`` is injected so the curvature
    logic can be unit tested with a fake.
    """

    def __init__(
        self,
        model,
        tokenizer,
        perturber: Perturber,
        n_perturbations: int = DEFAULT_N_PERTURBATIONS,
        max_length: int | None = None,
    ):
        if n_perturbations < 1:
            raise ValueError(f"n_perturbations must be >= 1, got {n_perturbations}")
        self.model = model
        self.tokenizer = tokenizer
        self.perturber = perturber
        self.n_perturbations = n_perturbations
        self.max_length = max_length

    @torch.no_grad()
    def _log_likelihood(self, text: str) -> float:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=self.max_length is not None,
            max_length=self.max_length,
        )
        device = getattr(self.model, "device", None)
        input_ids = enc["input_ids"].to(device) if device is not None else enc["input_ids"]

        logits = self.model(input_ids).logits[0]
        return mean_log_likelihood(logits, input_ids[0])

    def score(self, text: str) -> float:
        """Probabilistic-variation score for a single text."""
        original_ll = self._log_likelihood(text)
        neighbours = self.perturber.perturb(text, self.n_perturbations)
        perturbed_lls = [self._log_likelihood(t) for t in neighbours]
        return probabilistic_variation(original_ll, perturbed_lls)

    def score_batch(self, texts: list[str]) -> list[float]:
        """Probabilistic-variation score for each text, computed one at a time."""
        return [self.score(text) for text in texts]
