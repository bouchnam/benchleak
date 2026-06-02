"""Paraphrasers that generate "neighbour" texts for probabilistic variation.

The SFT detector measures how far a text's log-likelihood peaks above that of
slightly paraphrased versions of itself. These classes produce those paraphrases.
Both satisfy the ``Perturber`` protocol in ``detectors/sft.py``.

``WordSwapPerturber`` is a lightweight, dependency-free approximation that needs no
extra model. ``T5MaskFillPerturber`` follows the paper's semantic-domain default
(mask a fraction of tokens, refill with T5) for a higher-quality signal at the
cost of loading a second model.
"""

from __future__ import annotations

import random


class WordSwapPerturber:
    """Replace a fraction of words with other words drawn from the same text.

    A cheap, model-free paraphraser: it keeps length and vocabulary roughly intact
    while nudging the text off its exact form. Lower fidelity than a learned
    paraphraser, but it runs anywhere and is deterministic given ``seed``.
    """

    def __init__(self, fraction: float = 0.2, seed: int = 0):
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        self.fraction = fraction
        self._rng = random.Random(seed)

    def perturb(self, text: str, n: int) -> list[str]:
        words = text.split()
        if len(words) < 2:
            return [text] * n  # nothing meaningful to swap

        num_swaps = max(1, round(len(words) * self.fraction))
        neighbours = []
        for _ in range(n):
            swapped = words[:]
            for pos in self._rng.sample(range(len(swapped)), min(num_swaps, len(swapped))):
                swapped[pos] = self._rng.choice(words)
            neighbours.append(" ".join(swapped))
        return neighbours


class T5MaskFillPerturber:
    """Paraphrase by masking spans of words and refilling them with T5.

    Mirrors the paper's semantic-domain paraphraser: mask roughly ``mask_fraction``
    of the text as short spans, then let a T5 model predict replacements. The model
    is loaded lazily on first use so importing benchleak stays cheap. If a fill
    cannot be parsed, that neighbour falls back to the original text.
    """

    def __init__(
        self,
        model_name: str = "t5-base",
        mask_fraction: float = 0.2,
        span_length: int = 2,
        seed: int = 0,
        device: str | None = None,
        max_new_tokens: int = 200,
    ):
        if not 0 < mask_fraction <= 1:
            raise ValueError(f"mask_fraction must be in (0, 1], got {mask_fraction}")
        self.model_name = model_name
        self.mask_fraction = mask_fraction
        self.span_length = max(1, span_length)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._rng = random.Random(seed)
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is None:
            from transformers import AutoTokenizer, T5ForConditionalGeneration  # deferred

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self._model.eval()
            if self.device is not None:
                self._model.to(self.device)

    def _mask(self, words: list[str]) -> tuple[str, int]:
        """Replace random spans with T5 sentinels, returning the masked text and span count."""
        n_spans = max(1, int(len(words) * self.mask_fraction / self.span_length))
        masked = words[:]
        starts = self._rng.sample(range(len(words)), min(n_spans, len(words)))
        # Splice spans from the back so earlier indices stay valid as we replace.
        for start in sorted(starts, reverse=True):
            end = min(start + self.span_length, len(masked))
            masked[start:end] = ["<extra_id_placeholder>"]  # renumbered below
        # Renumber sentinels left-to-right as T5 expects (<extra_id_0>, <extra_id_1>, ...).
        out, sentinel = [], 0
        for token in masked:
            if token == "<extra_id_placeholder>":
                out.append(f"<extra_id_{sentinel}>")
                sentinel += 1
            else:
                out.append(token)
        return " ".join(out), sentinel

    def _fills(self, generated: str, n_spans: int) -> list[str]:
        """Split T5 output back into per-sentinel fill strings."""
        fills = []
        for i in range(n_spans):
            start = generated.find(f"<extra_id_{i}>")
            if start < 0:
                return []
            start += len(f"<extra_id_{i}>")
            end = generated.find(f"<extra_id_{i + 1}>")
            fill = generated[start:end if end >= 0 else None]
            fills.append(fill.replace("</s>", "").replace("<pad>", "").strip())
        return fills

    def _apply(self, masked: str, fills: list[str]) -> str:
        for i, fill in enumerate(fills):
            masked = masked.replace(f"<extra_id_{i}>", fill, 1)
        return masked

    def perturb(self, text: str, n: int) -> list[str]:
        import torch

        self._ensure_loaded()
        words = text.split()
        if len(words) < 2:
            return [text] * n

        neighbours = []
        for _ in range(n):
            masked, n_spans = self._mask(words)
            enc = self._tokenizer(masked, return_tensors="pt")
            device = getattr(self._model, "device", None)
            input_ids = enc["input_ids"].to(device) if device is not None else enc["input_ids"]
            with torch.no_grad():
                out = self._model.generate(input_ids, max_new_tokens=self.max_new_tokens, do_sample=True)
            generated = self._tokenizer.decode(out[0], skip_special_tokens=False)
            fills = self._fills(generated, n_spans)
            neighbours.append(self._apply(masked, fills) if fills else text)
        return neighbours
