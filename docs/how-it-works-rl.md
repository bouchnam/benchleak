# How the RL detector works

This document explains the RL-post-training contamination detector. For the
pre-training detector and the shared reference-set machinery, see
[how-it-works-pretrain.md](how-it-works-pretrain.md); for the fine-tuning
detector, see [how-it-works-sft.md](how-it-works-sft.md).

## Why pre-training and SFT detectors fail on RL

Both Min-K% (pre-training) and SPV-MIA (SFT) rely on a *likelihood* signal: a
memorised sample sits at high probability / low perplexity, because both training
phases are maximum-likelihood objectives that directly push up the probability of
seen text.

RL post-training (RLVR, GRPO) is different. It optimises an external *reward* —
typically 1 for a correct final answer, 0 otherwise — not the log-likelihood of a
ground-truth response. This decouples the model's behaviour from likelihood, so a
sample the model was rewarded on during RL leaves no low-perplexity fingerprint.
Tao et al. (2025, arXiv:2510.09259) show every likelihood-based detector drops to
roughly random (AUC ≈ 0.5) on RL-phase contamination.

## The signal: policy collapse, exposed by self-critique

RL narrows the model's search space to maximise pass@1: for problems it trained
on, the policy converges to a single highly-rewarded reasoning path (*policy
collapse*), visible as a sparse, near-deterministic token-level entropy curve.

The catch is that policy collapse also happens on *clean* problems, so raw entropy
alone is not a membership signal. Self-Critique adds an **active probe**: it asks
the model to redo the problem along a *different* reasoning path. A contaminated
problem is locked onto its rewarded trajectory and barely deviates, so the entropy
curves of the two responses stay highly similar. A clean problem still has
flexibility, so the curves diverge.

## The procedure

For a problem `q`, with greedy decoding throughout (the paper's best-performing
choice, their Appendix C.2):

1. **Initial response.** Generate `r1 = M(T(q))` and record its per-step
   token-level entropy `E1 = {H_t}`, where
   `H_t = -Σ_v p(v | x_<t) log p(v | x_<t)` is the entropy of the next-token
   distribution at decoding step `t`.
2. **Self-critique response.** Append a critique instruction containing `r1` to
   the question (`q' = q ⊕ I_critique(r1)`), generate `r2 = M(T(q'))`, and record
   `E2`.
3. **Score.** A length-penalised cosine similarity between the two entropy
   sequences:

   ```
   score(q) = cos(pad(E1), pad(E2)) × min(|E1|,|E2|) / max(|E1|,|E2|)
   ```

   The shorter sequence is zero-padded so position `i` is compared with position
   `i`; the length ratio penalises cases where one response is much longer, since
   that itself signals a different reasoning mode. A higher score means the model
   stayed on its path despite being told to change it — the memorisation
   signature. This matches the "higher = more contaminated" direction of the other
   detectors, so the same reference-based `scan` aggregates it into an AUC and
   p-value.

The math lives in `benchleak/detectors/rl.py`: the pure `step_entropy` and
`penalized_cosine_similarity`, kept separate from generation so they are
unit-tested by hand. The critique prompt is the paper's "Prompt 2 (Original)" from
Appendix E; their Table 9 shows the method is robust to paraphrases of it.

## A note on the input

Unlike the other two detectors, the input text is the *problem to solve*, not a
passage whose likelihood is scored: this detector **generates** from it. For a
benchmark like GSM8K, each sample's question (and answer) is fed as the problem.
The reference set plays the same calibration role as before.

## One departure from the paper

The paper thresholds the similarity score directly to decide membership. Following
the same choice as benchleak's SFT detector, we instead calibrate the score
against a reference *dataset* in `core.scan`, so the RL detector reports the same
AUC / p-value verdict as the other two rather than needing a tuned threshold.

## Caveats

- **Cost.** Two generations per sample makes this the slowest detector by far. Use
  a small `--limit`, tune `--max-new-tokens` (default 256; the paper uses far
  longer reasoning traces), and run on a GPU.
- **Needs a chat-templated model.** The probe assumes an instruct/RL-tuned model
  that follows the self-critique instruction; on a base model without a chat
  template it falls back to raw prompting and the signal is unreliable.
- **Reference confound.** As with the other detectors, the bundled general-prose
  reference confounds domain with memorisation against a narrow reasoning
  benchmark. Prefer a domain-matched `--reference` for results you intend to trust.

## Reference

Tao et al., 2025. *Detecting Data Contamination from Reinforcement Learning
Post-training for Large Language Models* (Self-Critique), ICLR 2026,
arXiv:2510.09259.
