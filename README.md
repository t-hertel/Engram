# Conditional Memory for Small Language Models: An Empirical Evaluation of N-Gram Hash Lookup Tables

*A Negative Result*

---

## TL;DR

DeepSeek's Engram modules (N-gram hash lookup tables) **reduce perplexity by 22-30%** on held-out text but produce **zero improvement on downstream benchmarks** (HellaSwag, PIQA, ARC-Challenge) across four experiments on models from 0.5B to 4B parameters on a single RTX 4090.

---

## What is Engram?

[Engram](https://arxiv.org/abs/2412.19437) is a conditional memory mechanism from DeepSeek-V3 that injects trainable embedding tables into transformer layers. Each table is indexed by a rolling N-gram hash of recent tokens, giving the model direct access to token-sequence statistics without the attention mechanism having to learn them from scratch.

It works beautifully at 671B parameters. The question we asked: **does it help at consumer hardware scale?**

## Results

| Model | Mode | Base PPL | Engram PPL | PPL Delta | HellaSwag | PIQA | ARC-C |
|:------|:-----|:--------:|:----------:|:---------:|:---------:|:----:|:-----:|
| Qwen2.5-0.5B | Frozen, 100K | 20.75 | 15.73 | **-24.2%** | +0.35 | -1.14 | +0.34 |
| Qwen2.5-1.5B | Frozen, 5K | 14.84 | 11.02 | **-25.7%** | +0.30 | n/a | -0.61 |
| Qwen3.5-4B | Frozen, 10K | 12.87 | 9.76 | **-24.2%** | -0.02 | -0.33 | +1.79 |
| Qwen3.5-2B | Full FT, 10K | 16.31 | 11.44 | **-29.9%** | +0.17 | -0.71 | -0.85 |

All downstream deltas are within noise (±2 pp). No consistent directional trend.

The 2B full fine-tuning experiment is the key result: the model co-adapts with Engram, achieves the **largest** perplexity reduction of any experiment, and still shows zero downstream improvement. This rules out "frozen model ignores the signal" as an explanation.

## Why Doesn't Perplexity Transfer?

We evaluate four hypotheses in the paper. The most compelling explanation: Engram captures local token co-occurrence statistics that are highly effective for next-token prediction but orthogonal to the multi-step reasoning and world knowledge that benchmarks require. A single Engram layer at position 1 captures 86% of the total perplexity gain, consistent with a surface-level statistical correction rather than deep representational learning.

## Repo Structure

```
engram_qwen35_integration.py   # Engram module (injects into Qwen3.5 models)
train_qwen35.py                # Train with frozen base model
train_qwen35_fullft.py         # Train with full joint fine-tuning
eval_qwen35.py                 # Perplexity evaluation (WikiText-103 test split)
bench_qwen35.py                # Downstream benchmarks (HellaSwag, PIQA, ARC-C)

results/
  consolidated_results.json    # All experiment results (machine-readable)

paper.tex                      # Paper source (LaTeX)
paper.md                       # Paper in Markdown
paper.pdf                      # Paper (PDF)
figure1_training_loss.png      # Training loss curve (Figure 1)
```

## Reproducing the Results

**Requirements:**
- Python 3.10+
- PyTorch (with CUDA)
- transformers, datasets, lm-eval
- bitsandbytes
- Single GPU with 24 GB VRAM (tested on RTX 4090)

**Quick start:**
```bash
pip install torch transformers datasets lm-eval bitsandbytes
```

**Run an experiment (frozen base + Engram):**
```bash
python train_qwen35.py --model Qwen/Qwen3.5-4B-Base --layers 1,8,16,23 --steps 10000
```

**Evaluate perplexity:**
```bash
python eval_qwen35.py --model Qwen/Qwen3.5-4B-Base --checkpoint experiments/qwen35_4b_10k
```

**Run downstream benchmarks:**
```bash
python bench_qwen35.py --model Qwen/Qwen3.5-4B-Base --checkpoint experiments/qwen35_4b_10k
```

**Full fine-tuning:**
```bash
python train_qwen35_fullft.py --model Qwen/Qwen3.5-2B-Base --layers 1,6,12,17 --steps 10000
```

> **Note:** You'll need to adjust HF model paths and data directories to match your setup. See script headers for full argument lists.

## Key Findings

1. **Perplexity is not a proxy for downstream performance.** A 30% perplexity reduction corresponded to literally zero downstream improvement.
2. **Frozen vs. full fine-tuning doesn't matter.** The largest PPL improvement (full FT, -29.9%) still produced zero benchmark gains.
3. **Scale matters.** Engram may genuinely help at 671B scale (DeepSeek-V3) but the effect does not emerge at 0.5B-4B.
4. **Local statistics ≠ reasoning.** Engram captures token co-occurrence patterns that help fill in blanks but don't enhance multi-step reasoning.

## Hardware

All experiments ran on a single NVIDIA RTX 4090 (24 GB VRAM):
- 4B frozen: fits easily (~10 GB VRAM)
- 2B full fine-tune: 14.08 GB peak VRAM
- 4B full fine-tune: OOM (requires >24 GB)

Engram embedding tables are stored on disk via mmap (~2.5 GB per layer, fp16) and only accessed pages are loaded into RAM.

## Paper

The full paper is available as [paper.pdf](paper.pdf), [paper.md](paper.md), or [paper.tex](paper.tex).

## Citation

```bibtex
@misc{engram_consumer_hw_2026,
  title={Conditional Memory for Small Language Models: An Empirical Evaluation of N-Gram Hash Lookup Tables},
  author={Independent Replication Study},
  year={2026}
}
```

## Acknowledgments

This work builds on [DeepSeek's Engram mechanism](https://github.com/deepseek-ai/Engram) and uses models from the [Qwen](https://github.com/QwenLM) family.

## License

Apache 2.0 (see [LICENSE](LICENSE)).
