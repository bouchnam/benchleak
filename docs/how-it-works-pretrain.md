# How the pre-training detector works

This document explains the pre-training contamination detector (Min-K% Prob) and
the shared reference-set machinery all three detectors build on: what it measures,
why it needs a reference set, how the verdict is computed, and where the method
can mislead. For the other detectors, see
[how-it-works-sft.md](how-it-works-sft.md) and
[how-it-works-rl.md](how-it-works-rl.md).

## The problem

When a model scores well on a benchmark like GSM8K, there are two explanations:
the model genuinely generalises, or it saw the benchmark during training and is
partly recalling it. A leaderboard number alone cannot tell these apart.
benchleak attacks this with a membership-inference test: it asks whether the
model's own token probabilities betray that it has seen the benchmark text
before.

## The signal: Min-K% Prob

The detector implements Min-K% Prob (Shi et al., 2024). The hypothesis, quoted
from the paper:

> a non-member example is more likely to include a few outlier words with high
> negative log-likelihood (or low probability), while a member example is less
> likely to include such words.

In plain terms: a text the model was trained on holds few surprises, even on its
rare words. A text the model has never seen tends to contain a handful of tokens
it finds genuinely improbable.

### How a single text is scored

For a text with tokens `x_1 ... x_N`:

1. Run one forward pass of the model over the whole text. Because of causal
   masking, this yields, at every position, the probability the model assigns to
   the token that actually comes next. No text is generated; we read off the
   probability of the real next token (`token_log_probs` in
   `benchleak/detectors/pretrain.py`).
2. This gives one log-probability per token: how unsurprised the model was by
   each real token.
3. Discard the easy tokens. Keep only the lowest `k%` (default 20%), the tokens
   the model found least likely, and average their log-probabilities
   (`min_k_prob`).

The easy tokens (spaces, common words) look the same in every text and carry no
signal, so the bottom `k%` is where the memorisation evidence concentrates.

A single score is a log-probability, so it is negative. Reading the direction:

- Higher (closer to zero): the model was barely surprised, even by the hardest
  tokens. The text looks trained on.
- Lower (more negative): the model tripped on rare tokens. The text looks unseen.

## Why a reference set is needed

A raw Min-K% score is uninterpretable on its own. Its scale depends on the
model, the tokenizer, the text length, and the domain. A score of -5.3 means
nothing in isolation; there is no universal contaminated threshold.

The fix is a control group. We score a second set of texts the model is known
not to have trained on (the **reference set**), and ask whether the benchmark
scores separate from it. The benchmark is the suspect group; the reference is the
known-clean control. Without the control, no conclusion is possible.

### "Reference-free" in the paper means something different

The paper titles its method "reference-free", which can cause confusion. There
"reference" means a second **model** used by earlier methods to calibrate
difficulty:

> [prior methods] typically use reference models to compute the background
> difficulty of the data point and to calibrate the output probability of the
> target language model.

Min-K% deliberately avoids needing a calibration model. That is not the kind of
reference benchleak uses. Our reference is **data**, not a model: a set of
non-member texts. The paper relies on exactly this kind of data too. Its WikiMIA
benchmark is built from text published after the model's cutoff:

> We construct our benchmark by using events added to Wikipedia after specific
> dates, treating them as non-member data since they are guaranteed not to be
> present in the pretraining data.

And to make a real yes/no call in its GPT-3 case study, the paper learns a
threshold from labelled examples ("50 books known to be memorized"). So the need
for a clean baseline is real and present in the paper; benchleak simply packages
it as a per-benchmark comparison rather than a hand-tuned threshold.

### The bundled reference and its limitation

A small default reference ships in `benchleak/data/reference.txt`: original prose
passages written for this project, so no released model can have trained on them.
This makes the tool run out of the box.

The limitation: that default is general prose, while a benchmark like GSM8K is
math. The separation then mixes two effects, only one of which we want:

1. seen vs unseen (the contamination signal)
2. math vs prose (an accidental domain difference; math is inherently more
   predictable to an LLM)

For a result you intend to trust, supply your own reference with `--reference`:
text in the **same domain** as the benchmark but **guaranteed unseen** (for
example, math word problems published after the model's training cutoff). Then
the only remaining difference between the two groups is whether the model saw
them, which is exactly what the test should isolate.

## From scores to a verdict

After scoring, we have two sets of numbers (benchmark and reference) and ask
whether the benchmark scores are systematically higher.

### AUC

The AUC is the fraction of all (benchmark, reference) pairs in which the
benchmark score is higher, with ties counting as half. With 25 samples per side
that is 625 pairs. An AUC of 0.5 means the two sets are indistinguishable; 1.0
means every benchmark sample outscores every reference sample. The Mann-Whitney U
statistic yields this directly: `AUC = U / (n_benchmark * n_reference)`
(`compare_distributions` in `benchleak/core.py`).

### p-value and why Mann-Whitney, not a t-test

The p-value is the probability of seeing separation this strong if the two sets
actually came from the same distribution. We use the one-sided Mann-Whitney U
test (Wilcoxon rank-sum) rather than a t-test because:

- It assumes no particular distribution shape. Min-K% scores are skewed, so a
  t-test's normality assumption would make its p-value unreliable. Mann-Whitney
  operates on ranks.
- It tests the right hypothesis: whether benchmark scores tend to exceed
  reference scores (stochastic dominance), not whether the two means are equal.
- Its U statistic is the AUC, so one test gives both the effect size and the
  significance.
- It is robust to outliers, which matters at small sample sizes.

One caveat of a rank-based test: on very small samples the p-value has a floor.
With 3 samples per side it can never drop below 0.05, so a trustworthy verdict
needs at least 5 samples per side.

### The decision

A benchmark is flagged as likely contaminated only when both gates pass
(`ContaminationResult.contaminated`):

```
AUC >= 0.60   AND   p < 0.05
```

The AUC gate requires the effect to be large enough to matter; the p-value gate
requires it to be unlikely to be noise. Both are deliberately conservative
defaults, not hard science, and can be adjusted.

## Summary of caveats

- A high AUC against the bundled prose reference reflects domain as well as
  memorisation. Use a domain-matched `--reference` before trusting a verdict.
- A verdict needs at least 5 samples per side.
- The thresholds (AUC 0.6, p 0.05) are heuristics, not calibrated guarantees.
- Only the pre-training phase is implemented. A clean pre-training result does
  not rule out contamination introduced during SFT or RL.

## References

- Shi et al., 2024. *Detecting Pretraining Data from Large Language Models.*
  arXiv:2310.16789. The Min-K% Prob method and the WikiMIA benchmark.
- Fu et al., 2024. *Membership Inference via Self-Prompt Calibration.* Basis for
  the planned SFT detector.
- Tao et al., 2025. *Detecting Data Contamination from RL Post-training.*
  arXiv:2510.09259. Basis for the planned RL detector.
