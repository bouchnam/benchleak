"""Min-K% Prob pre-training contamination detector (Shi et al. 2024).

Reference-free membership inference. The hypothesis: a text seen during
pre-training rarely contains tokens the model finds surprising, so the mean
log-probability of its *least* likely tokens stays comparatively high. An unseen
text, by contrast, tends to carry a few low-probability outlier tokens that drag
that mean down.

For tokens x = x_1 ... x_N, with per-token log-likelihood log p(x_i | x_<i), we
keep the k% of tokens with the lowest probability (the set Min-K%(x)) and average
their log-likelihood:

    Min-K%-Prob(x) = (1 / |Min-K%(x)|) * sum_{x_i in Min-K%(x)} log p(x_i | x_<i)

A higher score means the text is more likely to have been in the pre-training
data. Membership is decided by thresholding the score.

Reference: "Detecting Pretraining Data from Large Language Models", ICLR 2024
(arXiv:2310.16789).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

DEFAULT_K = 20.0


def token_log_probs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Log-probability the model assigns to each realised next token.

    ``logits`` has shape ``(seq_len, vocab)`` where ``logits[i]`` scores the token
    following position ``i``. Returns a ``(seq_len - 1,)`` tensor holding
    ``log p(x_{i+1} | x_<=i)`` for every position with a successor.
    """
    if logits.ndim != 2:
        raise ValueError(f"expected logits of shape (seq_len, vocab), got {tuple(logits.shape)}")
    if input_ids.ndim != 1:
        raise ValueError(f"expected input_ids of shape (seq_len,), got {tuple(input_ids.shape)}")
    if logits.shape[0] != input_ids.shape[0]:
        raise ValueError("logits and input_ids must share the same sequence length")

    log_probs = F.log_softmax(logits[:-1], dim=-1)
    targets = input_ids[1:]
    return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def min_k_prob(token_lps: torch.Tensor, k: float = DEFAULT_K) -> float:
    """Mean log-probability of the lowest-``k``% tokens.

    ``token_lps`` is a 1-D tensor of per-token log-probabilities. ``k`` is a
    percentage in ``(0, 100]``. At least one token is always retained.
    """
    if not 0 < k <= 100:
        raise ValueError(f"k must be a percentage in (0, 100], got {k}")
    n = token_lps.numel()
    if n == 0:
        raise ValueError("cannot score an empty sequence of token log-probabilities")

    num_kept = max(1, math.floor(n * k / 100))
    lowest = torch.topk(token_lps, num_kept, largest=False).values
    return lowest.mean().item()


class MinKProbDetector:
    """Score texts for pre-training membership with Min-K% Prob.

    The detector is agnostic to where the model comes from: any object exposing a
    ``transformers``-style ``model(input_ids).logits`` forward pass and a matching
    tokenizer will do, which keeps it cheap to unit test.
    """

    def __init__(self, model, tokenizer, k: float = DEFAULT_K, max_length: int | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.k = k
        self.max_length = max_length

    @torch.no_grad()
    def score(self, text: str) -> float:
        """Min-K% Prob score for a single text."""
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=self.max_length is not None,
            max_length=self.max_length,
        )
        device = getattr(self.model, "device", None)
        input_ids = enc["input_ids"].to(device) if device is not None else enc["input_ids"]

        logits = self.model(input_ids).logits[0]
        token_lps = token_log_probs(logits, input_ids[0])
        return min_k_prob(token_lps, self.k)

    def score_batch(self, texts: list[str]) -> list[float]:
        """Min-K% Prob score for each text, computed one at a time."""
        return [self.score(text) for text in texts]
