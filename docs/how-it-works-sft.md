# How the SFT detector works

This document explains the fine-tuning (SFT) contamination detector. For the
pre-training detector and the shared reference-set machinery, see
[how-it-works-pretrain.md](how-it-works-pretrain.md).

## The problem with probability as a signal

The pre-training detector (Min-K% Prob) assumes a memorised text has high
probability. Fu et al. (2024, arXiv:2311.06062, "SPV-MIA") point out that this
only holds when the model is *overfit*. Fine-tuning with regularisation and early
stopping avoids overfitting, yet the model still *memorises* its training data.
So a probability-based signal produces many false positives on realistically
fine-tuned models.

## The signal: probabilistic variation

SPV-MIA replaces "is the probability high?" with "**does the text sit on a local
peak of the probability landscape?**" Memorisation makes a fine-tuned-on example a
local maximum: nudging the text slightly lowers its probability. An unseen text
sits on a flat slope, so nudging barely changes it.

The paper formalises this as the curvature (second derivative) of the probability,
approximated with perturbations (Eqs. 8 to 10). In its practical neighbour form:

```
pv(x) = loglik(x) - mean_n loglik(paraphrase_n)
```

where `loglik` is the mean per-token log-likelihood, and the paraphrases are
slight rewrites of `x`. A fine-tuned-on text is a peak, so its log-likelihood
exceeds its paraphrases and `pv(x)` is large and positive. An unseen text gives
`pv(x)` near zero. Higher score means more likely memorised, the same direction as
Min-K%, so the same reference-based `scan` aggregates it into an AUC and p-value.

The math lives in `benchleak/detectors/sft.py`: `mean_log_likelihood` and the pure
`probabilistic_variation`, kept separate from model I/O so they are unit-tested by
hand.

## Paraphrasers

The neighbours are produced by a pluggable `Perturber`
(`benchleak/detectors/perturb.py`):

- `T5MaskFillPerturber` (default, `--perturber t5`): masks a fraction of the text
  as short spans and refills them with a T5 model. This follows the paper's
  semantic-domain default and gives the cleaner signal, at the cost of downloading
  a second model.
- `WordSwapPerturber` (`--perturber word`): replaces a fraction of words with other
  words from the same text. No extra model and runs anywhere, but lower fidelity;
  random swaps are a weak approximation of a learned paraphraser.

## Two departures from the paper

1. **No self-prompt reference model.** SPV-MIA also fine-tunes a reference model on
   text the target model generates, to calibrate the signal (the "self-prompt
   calibration" the title refers to). Fine-tuning a model locally is impractical
   for a tool meant to run in minutes, so benchleak instead calibrates with its
   reference *dataset* comparison in `core.scan`, exactly as the pre-training
   detector does.
2. **Asymmetric neighbour form.** The paper's symmetric second-derivative estimate
   is reduced to the neighbour form it shows it rediscovers, which needs only
   forward paraphrases rather than matched symmetric pairs.

## Caveats

- Signal quality depends heavily on the paraphraser. Prefer `--perturber t5`, and
  a domain-matched `--reference`, for results you intend to trust.
- Cost: each sample needs one forward pass for the original plus one per
  paraphrase (`--n-perturbations`, default 10), so this detector is roughly an
  order of magnitude slower than the pre-training detector. Use a small `--limit`
  and a GPU where possible.
- As with the pre-training detector, a raw score is only meaningful relative to the
  reference set, and the bundled prose reference confounds domain with
  memorisation against a narrow benchmark.

## Reference

Fu et al., 2024. *Membership Inference Attacks against Fine-tuned Large Language
Models via Self-prompt Calibration* (SPV-MIA), NeurIPS 2024, arXiv:2311.06062.
