# RoPE Attention Benchmark

A bare-bones, single-file benchmark that compares scaled dot-product attention
**with** and **without** Rotary Positional Encoding (RoPE).

"Is RoPE worth it?" has two answers, so the script measures two things:

1. **Latency** — how much wall-clock overhead RoPE actually adds to an attention
   forward pass, swept across a range of sequence lengths.
2. **Behavior** — *why* RoPE earns that overhead. It demonstrates the
   relative-position property (attention logits depend only on `i - j`, not on
   absolute position) and contrasts it with plain attention, which is
   permutation-equivariant and therefore blind to word order entirely.

Everything is written from scratch on purpose — no `nn.MultiheadAttention`, no
flash kernels. The point is to *see* the mechanism, not hide it. The only
dependency is `torch`.

---

## Quick start

```bash
# from this directory
python3 -m venv .venv
source .venv/bin/activate
pip install torch
python rope_attention_benchmark.py
```

On Apple Silicon, use the Homebrew Python to build the venv
(`/opt/homebrew/bin/python3 -m venv .venv`) — the system/pyenv Python can be
architecture-broken. The script auto-selects CUDA, then MPS, then CPU.

> If you see a `Failed to initialize NumPy` warning, it's harmless — torch just
> probes for numpy. `pip install numpy` silences it.

---

## Usage

```bash
python rope_attention_benchmark.py                          # defaults
python rope_attention_benchmark.py --device cuda --seq-lens 256 1024 4096
python rope_attention_benchmark.py --csv results.csv        # also write CSV
python rope_attention_benchmark.py --no-color               # plain text, for logs
python rope_attention_benchmark.py --no-demo                # latency only
```

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--batch` | `8` | Batch size |
| `--heads` | `8` | Number of attention heads |
| `--head-dim` | `64` | Per-head dimension (must be even for RoPE) |
| `--seq-lens` | `128 256 512 1024 2048` | Sequence lengths to sweep |
| `--causal` | off | Apply a causal mask |
| `--warmup` | `10` | Warmup iterations (discarded) |
| `--iters` | `50` | Timed iterations (median reported) |
| `--base` | `10000.0` | RoPE frequency base (θ) |
| `--device` | `auto` | `auto` \| `cpu` \| `cuda` \| `mps` |
| `--dtype` | `float32` | `float32` \| `float16` \| `bfloat16` |
| `--seed` | `0` | RNG seed |
| `--csv` | — | Path to write results as CSV |
| `--no-demo` | off | Skip the behavior demo |
| `--no-color` | off | Disable ANSI colors |

---

## Reading the output

### Part 1 — Latency

For each sequence length, the script times attention twice on identical random
`q/k/v` tensors: once plain, once with RoPE applied to `q` and `k`. It reports
the **median** time per forward pass (median, not mean, so a single OS hiccup
doesn't move the number) and the extra cost RoPE adds.

```
 seq_len |   no-PE (ms) |    RoPE (ms) |   overhead
---------------------------------------------------
     512 |        1.267 |        1.360 |       7.4%
    2048 |       22.264 |       26.329 |      18.3%
```

The **overhead** column is color-coded: green `< 10%`, yellow `< 30%`, red
beyond. RoPE's cost is a fixed `O(seq · head_dim)` elementwise pass, so in
*relative* terms it shrinks as the `O(seq²)` attention matmul comes to dominate.

### Part 2 — Behavior

Four labeled checks that explain the value RoPE provides:

1. **Relative-position invariance** — slide a fixed `(query, key)` pair right by
   the same offset and the score stays put. RoPE encodes only the gap `i - j`.
2. **Relative position is visible** — different gaps to the same query produce
   genuinely different scores.
3. **No PE → order is invisible** — permuting the keys just permutes the logit
   columns identically. Raw attention is permutation-equivariant, so
   *"dog bites man"* and *"man bites dog"* look identical to it.
4. **Sanity check** — the hand-written SDPA matches torch's reference kernel to
   machine precision (`PASS`).

The behavior demo runs in `float64` for tight tolerances. Since MPS has no
float64 path, on MPS devices this part transparently falls back to the CPU — it
is tiny and not performance-sensitive.

---

## Files

- [`rope_attention_benchmark.py`](rope_attention_benchmark.py) — the entire suite
  (RoPE, attention, timing harness, behavior demo, and CLI).

## How RoPE works here

This uses the **rotate-half** formulation (LLaMA and most modern codebases): it
pairs dimension `p` with `p + d/2` and rotates each pair by an angle
proportional to the token's position. That's mathematically equivalent (up to a
permutation of dims) to the adjacent-pair formulation in the
[original paper](https://arxiv.org/abs/2104.09864). RoPE is applied to the
queries and keys only, never the values.
