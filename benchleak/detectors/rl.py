"""Self-Critique RL post-training contamination detector (Tao et al. 2025).

Implements Self-Critique from "Detecting Data Contamination from Reinforcement
Learning Post-training for Large Language Models" (ICLR 2026, arXiv:2510.09259).

Why a new detector is needed: pre-training and SFT are maximum-likelihood
objectives, so a memorised sample carries a low-perplexity fingerprint that
Min-K% (pretrain) and SPV-MIA (sft) read off. RL post-training (RLVR/GRPO)
optimises a *reward* instead of likelihood, leaving no such fingerprint, so
likelihood-based detectors collapse to near-random on RL-phase contamination.

The signal Self-Critique uses is *policy collapse*. After RL, a model converges
to a narrow, highly-rewarded reasoning path for the problems it trained on. Raw
output entropy alone does not reveal this (clean samples also go low-entropy), so
the method probes *actively*: it asks the model to redo the problem along a
*different* reasoning path. A contaminated problem is locked onto its rewarded
trajectory and barely deviates, so the per-token entropy curves of the two
responses stay highly similar; a clean problem explores, so the curves diverge.

For a problem ``q`` (greedy decoding throughout):

    r1 = M(T(q))                      ; E1 = per-step entropy of r1
    r2 = M(T(q + I_critique(r1)))     ; E2 = per-step entropy of r2
    score(q) = cos(pad(E1), pad(E2)) * min(|E1|,|E2|) / max(|E1|,|E2|)

where per-step entropy is H_t = -sum_v p(v|x_<t) log p(v|x_<t). A higher score
means the model stayed on the same path despite being told to change it, i.e. the
memorisation signature, matching the "higher = more contaminated" convention of
the other detectors, so the same reference-based ``core.scan`` aggregates it.

Unlike the other two detectors, the input text here is the *problem* to solve, not
a passage to score: this detector *generates* from it. One deliberate departure
from the paper, in keeping with the SFT detector: the paper thresholds the score
directly, whereas benchleak calibrates it against a reference *dataset* in
``core.scan`` (see docs/how-it-works-rl.md).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

DEFAULT_MAX_NEW_TOKENS = 256

# Paper's "Prompt 2: Self-Critique Instruction (Original)" (Appendix E). The probe
# is robust to paraphrases of this (their Table 9), so the exact wording is not
# load-bearing, but this is the one their main results use.
CRITIQUE_TEMPLATE = (
    "A possible answer is provided below (it may or may not be correct). "
    "Please provide a response that follows a different reasoning path or "
    "provides an alternative solution:\n"
    "—\n"
    "{response}\n"
    "—\n"
    "Please now provide your new, different response:"
)


def step_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (in nats) of the next-token distribution at each step.

    ``logits`` has shape ``(n_steps, vocab)`` where each row is the model's
    pre-softmax scores for one decoding step. Returns a ``(n_steps,)`` tensor of
    entropies ``H = -sum_v p_v log p_v``. ``log_softmax`` keeps every probability
    strictly positive, so ``p * log p`` is finite and the sum is stable.
    """
    if logits.ndim != 2:
        raise ValueError(f"expected logits of shape (n_steps, vocab), got {tuple(logits.shape)}")

    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def penalized_cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Length-penalised cosine similarity between two entropy sequences.

    Position ``i`` of one response is compared with position ``i`` of the other;
    the shorter sequence is zero-padded so non-overlapping positions contribute
    nothing. The cosine is then scaled by ``min(len)/max(len)`` because a large
    length gap is itself a sign the two reasoning paths differ. Returns ``0.0`` if
    either sequence is empty or has zero norm (no usable signal).
    """
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    if not a or not b:
        return 0.0

    width = max(len(a), len(b))
    va = torch.tensor(a + [0.0] * (width - len(a)))
    vb = torch.tensor(b + [0.0] * (width - len(b)))
    norm = va.norm() * vb.norm()
    if norm == 0:
        return 0.0

    cosine = float(torch.dot(va, vb) / norm)
    length_penalty = min(len(a), len(b)) / max(len(a), len(b))
    return cosine * length_penalty


class SelfCritiqueDetector:
    """Score problems for RL-phase membership via self-critique entropy similarity.

    Like the other detectors, this is agnostic to the model's origin: any object
    exposing a ``transformers``-style ``generate`` plus a matching tokenizer works.
    The pure ``step_entropy`` and ``penalized_cosine_similarity`` carry the math so
    they can be unit-tested without a model.
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        critique_template: str = CRITIQUE_TEMPLATE,
        max_length: int | None = None,
    ):
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.critique_template = critique_template
        self.max_length = max_length

    def _format_prompt(self, user_text: str) -> str:
        """Wrap user text in the model's chat template when it has one.

        Self-Critique targets instruct/RL-tuned models, which expect a chat
        template; base models without one fall back to the raw text.
        """
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return user_text

    @torch.no_grad()
    def _generate(self, user_text: str) -> tuple[torch.Tensor, str]:
        """Greedy-decode a response, returning its entropy sequence and text."""
        prompt = self._format_prompt(user_text)
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=self.max_length is not None,
            max_length=self.max_length,
        )
        device = getattr(self.model, "device", None)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")
        if device is not None:
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        out = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,  # greedy: the paper's best-performing decoding (App. C.2)
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=pad_token_id,
        )
        if not out.scores:  # model emitted nothing past the prompt
            return torch.empty(0), ""

        logits = torch.stack(out.scores, dim=0)[:, 0, :]  # (n_steps, vocab)
        entropies = step_entropy(logits)
        response_ids = out.sequences[0, input_ids.shape[1]:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return entropies, response

    def score(self, text: str) -> float:
        """Self-Critique similarity score for a single problem."""
        e1, r1 = self._generate(text)
        critique = self.critique_template.format(response=r1)
        e2, _ = self._generate(f"{text}\n\n{critique}")
        return penalized_cosine_similarity(e1.tolist(), e2.tolist())

    def score_batch(self, texts: list[str]) -> list[float]:
        """Self-Critique score for each problem, computed one at a time."""
        return [self.score(text) for text in texts]
