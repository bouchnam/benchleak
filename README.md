# benchleak 🔍

> **Did this model train on the test set? Find out in one command.**

When a model scores 90% on GSM8K or MATH, was it genuinely capable or did it
memorise the benchmark during training? Benchleak answers that with a
mathematical membership-inference test on the model's own token probabilities.
No LLM judges, no API calls, runs locally on any HuggingFace causal LM.

## Status

🚧 **Early alpha.** The **pre-training** detector (Min-K% Prob) is implemented and
runnable end to end. The SFT and RL-post-training detectors are planned but **not
yet built**, so don't expect them yet.

## Install

```bash
pip install benchleak
```

This pulls in `torch`, `transformers`, and `datasets`. To work from a clone
instead, run `pip install -e .` in the repo root.

## Usage

```bash
benchleak --model Qwen/Qwen2.5-0.5B --benchmark gsm8k
```

The model can be any HuggingFace-format causal LM, given as a Hub id or a local
checkpoint directory. GGUF, llama.cpp, and Ollama formats are not supported.

### Private or gated models

For a repository that requires authentication, provide a HuggingFace token. Any
of these work:

```bash
huggingface-cli login                  # cached credential, picked up automatically
export HF_TOKEN=hf_xxx                  # environment variable
benchleak --model my/private-model --benchmark gsm8k --hf-token hf_xxx
```

### Speed

Scoring runs one forward pass per sample. On a CPU-only machine the default
`--limit 200` can take many minutes; use a smaller `--limit` for a quick look, or
`--device cuda` / `--device mps` to use a GPU.

```
benchleak: pre-training contamination report
====================================================
Model:       Qwen/Qwen2.5-0.5B
Benchmark:   gsm8k
Detector:    min-k% prob
Samples:     200 benchmark vs 200 reference

Separation (AUC):   0.71   [HIGH]
Significance (p):   3.2e-08
Flag thresholds:    AUC >= 0.6, p < 0.05

Verdict: LIKELY CONTAMINATED
```

Known benchmarks (`gsm8k`, `math`, `arc-challenge`, `truthfulqa`) work by name. For
any other Hub dataset, pass the path plus its text column(s):

```bash
benchleak --model my/model --benchmark some/dataset --field question --field answer
```

### Your own benchmark from a local file

Point `--benchmark` at a local file instead of a Hub id. Supported formats:

- `.txt`: one passage per line (no `--field` needed)
- `.jsonl` / `.json` / `.csv`: name the text column(s) with `--field`

```bash
benchleak --model my/model --benchmark ./my_benchmark.jsonl --field question --field answer
```

Local files are read with the standard library, so this path needs neither a
network connection nor the `datasets` package.

## How it works

The benchmark is scored against a **reference set** of text the model is not
expected to have memorised. Min-K% Prob assigns each text the mean log-probability
of its least-likely *k%* of tokens. Memorised text has fewer surprising tokens
and scores higher. The tool then measures how strongly the benchmark's scores separate
from the reference's, reported as an AUC (`U / nm` from a Mann-Whitney test) with
a significance p-value. AUC ≈ 0.5 means the benchmark looks like fresh data; AUC
well above 0.5 is the memorisation signature of contamination.

A small reference set ships with the tool so it runs out of the box. For the
cleanest signal, supply your own domain-matched reference data with
`--reference my_reference.txt` (one passage per line).

For the full reasoning (why a reference set is needed, the choice of test, and
where the method can mislead), see [docs/how-it-works.md](docs/how-it-works.md).

| Phase | Method | Status |
|-------|--------|--------|
| Pre-training | Min-K% probability (Shi et al. 2024) | ✅ implemented |
| SFT | Self-prompt calibration (Fu et al. 2024) | ⏳ planned |
| RL post-training | Self-Critique entropy (Tao et al. 2025) | ⏳ planned |

## Caveats

- A verdict needs **≥ 5 samples per side**; the significance test cannot reach
  p < 0.05 below that.
- The bundled reference is general-domain prose. Comparing it against a
  narrow-domain benchmark (e.g. math) can confound *domain* with *memorisation*;
  prefer a domain-matched `--reference` for results you intend to publish.

## Troubleshooting

**`ModuleNotFoundError: No module named '_lzma'`** when loading a benchmark. The
`datasets` library needs Python's `lzma` module, which is absent from some Python
builds (commonly pyenv on macOS compiled without the `xz` library). Install `xz`
and rebuild Python:

```bash
brew install xz
LDFLAGS="-L$(brew --prefix xz)/lib" CPPFLAGS="-I$(brew --prefix xz)/include" \
  pyenv install -f <your-python-version>
```

Then reuse or recreate your virtual environment. benchleak detects this case and
prints the same guidance.

## Citation

Implements methods from:
- Shi et al., 2024. *Detecting Pretraining Data from Large Language Models* (arXiv:2310.16789)
- Fu et al., 2024. *Membership Inference via Self-Prompt Calibration*
- Tao et al., 2025. *Detecting Data Contamination from RL Post-training* (arXiv:2510.09259)

## License

Apache-2.0
