# benchleak 🔍

> **Did this model cheat on this benchmark? Find out in one command.**

Benchleak detects benchmark contamination in LLMs across all three training phases — pre-training, SFT, and RL post-training.

When a model scores 90% on GSM8K or MATH, was it genuinely smart or did it train on the test set? Current leaderboards have no answer. Benchleak does.

## Install

```bash
pip install benchleak
```

## Usage

```bash
benchleak --model Qwen/Qwen2.5-7B --benchmark gsm8k
```

```
GSM8K Contamination Report — Qwen2.5-7B
─────────────────────────────────────────
Pre-training phase:     0.71 ⚠️  HIGH
SFT phase:              0.43    MEDIUM
RL phase:               0.89 🚨 VERY HIGH

Overall verdict: LIKELY CONTAMINATED
Confidence: 94%
```

## How it works

Three mathematically grounded detectors, one per training phase:

| Phase | Method | Signal |
|-------|--------|--------|
| Pre-training | Min-K% probability | Unnaturally low loss on benchmark tokens |
| SFT | Self-prompt calibration | Memorized surface form vs. paraphrase gap |
| RL post-training | Self-Critique entropy | Policy collapse on benchmark samples |

No LLM judges. No API calls. Pure math on model internals.

## Status

🚧 Under active development.

## Citation

This tool implements and unifies methods from:
- Shi et al., 2024 — *Detecting Pretraining Data from Large Language Models*
- Fu et al., 2024 — *Membership Inference via Self-Prompt Calibration*
- Tao et al., 2025 — *Detecting Data Contamination from RL Post-training*

## License

MIT
