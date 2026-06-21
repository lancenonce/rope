"""
rope_attention_benchmark.py

A bare-bones, single-file benchmark that compares scaled dot-product attention
WITH and WITHOUT Rotary Positional Encoding (RoPE).

It measures two different things, because "is RoPE worth it" has two answers:

  1. Latency. How much wall-clock overhead does RoPE actually add to an
     attention forward pass? Swept across a range of sequence lengths.

  2. Behavior. *Why* RoPE earns that overhead. We demonstrate the
     relative-position property (attention logits depend only on i - j, not on
     absolute position) and contrast it with plain attention, which is
     permutation-equivariant and therefore blind to order entirely.

Everything here is written from scratch on purpose. No nn.MultiheadAttention,
no flash kernels. The point is to see the mechanism, not to hide it.

Dependency: torch.

Usage:
    python rope_attention_benchmark.py
    python rope_attention_benchmark.py --device cuda --seq-lens 256 1024 4096
    python rope_attention_benchmark.py --csv results.csv
    python rope_attention_benchmark.py --no-color      # plain text, for logs
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


# ===========================================================================
# Tiny ANSI color layer.
#
# We deliberately avoid pulling in a dependency like `rich` or `colorama`.
# These are the only escape codes we need. Color is auto-disabled when output
# is not a TTY (piped to a file, captured by CI), when NO_COLOR is set, or when
# --no-color is passed. That keeps CSVs and log files clean.
# ===========================================================================
class C:
    """ANSI codes. Every attribute becomes an empty string when color is off."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GREY = "\033[90m"

    @classmethod
    def disable(cls) -> None:
        for name in dir(cls):
            if name.isupper():
                setattr(cls, name, "")


def _color_enabled(force_off: bool) -> bool:
    if force_off:
        return False
    if os.environ.get("NO_COLOR") is not None:
        return False
    return sys.stdout.isatty()


def rule(char: str = "=", width: int = 74, color: str = "") -> None:
    print(f"{color}{char * width}{C.RESET}")


def header(title: str) -> None:
    """A bold, boxed section banner."""
    rule("=", color=C.CYAN)
    print(f"{C.BOLD}{C.CYAN}{title}{C.RESET}")
    rule("=", color=C.CYAN)


def note(text: str) -> None:
    """Dim explanatory prose — the 'what is happening' narration."""
    print(f"{C.GREY}{text}{C.RESET}")


# ===========================================================================
# RoPE: the rotate-half formulation used by LLaMA / most modern codebases.
# It pairs dimension p with dimension p + d/2 and rotates each pair by an angle
# proportional to the token's position. Mathematically equivalent (up to a
# permutation of dims) to the adjacent-pair formulation in the original paper.
# ===========================================================================
def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Precompute cos and sin tables for RoPE.

    Returns two tensors of shape [seq_len, head_dim]. We build the angles in
    float32 for numerical stability, then cast to the working dtype.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")

    # One frequency per dimension pair: theta_i = base^(-2i / d).
    exponents = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
    inv_freq = 1.0 / (base ** exponents)                      # [head_dim/2]

    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)                  # [seq_len, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)                   # [seq_len, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: Tensor) -> Tensor:
    """Rotate the two halves of the last dimension: [a, b] -> [-b, a]."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to x.

    x is [batch, heads, seq, head_dim]; cos and sin are [seq, head_dim] and
    broadcast over batch and heads.
    """
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin


# ===========================================================================
# Bare-bones scaled dot-product attention.
# ===========================================================================
def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, causal: bool = False
) -> Tensor:
    """Manual SDPA. q, k, v are [batch, heads, seq, head_dim]."""
    head_dim = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)   # [b, h, s, s]

    if causal:
        seq = scores.shape[-1]
        mask = torch.triu(
            torch.ones(seq, seq, device=scores.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))

    attn = scores.softmax(dim=-1)
    return attn @ v                                            # [b, h, s, head_dim]


def attention_no_pe(q: Tensor, k: Tensor, v: Tensor, causal: bool = False) -> Tensor:
    """Attention with no positional information at all."""
    return scaled_dot_product_attention(q, k, v, causal=causal)


def attention_rope(
    q: Tensor, k: Tensor, v: Tensor, cos: Tensor, sin: Tensor, causal: bool = False
) -> Tensor:
    """Attention with RoPE applied to the queries and keys (never the values)."""
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    return scaled_dot_product_attention(q, k, v, causal=causal)


# ===========================================================================
# Timing harness.
# ===========================================================================
@dataclass
class BenchConfig:
    batch: int
    heads: int
    head_dim: int
    seq_lens: list[int]
    causal: bool
    warmup: int
    iters: int
    base: float
    device: torch.device
    dtype: torch.dtype
    seed: int = 0


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_fn(fn, warmup: int, iters: int, device: torch.device) -> float:
    """Return the median wall-clock time per call, in seconds.

    Median, not mean: a single OS hiccup or GC pause shouldn't move the number.
    Warmup runs are discarded so we don't time lazy allocation / autotuning.
    """
    for _ in range(warmup):
        fn()
    _sync(device)

    samples: list[float] = []
    for _ in range(iters):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        samples.append(time.perf_counter() - start)

    samples.sort()
    return samples[len(samples) // 2]


@dataclass
class Row:
    seq_len: int
    no_pe_ms: float
    rope_ms: float

    @property
    def overhead_pct(self) -> float:
        if self.no_pe_ms == 0:
            return float("nan")
        return 100.0 * (self.rope_ms - self.no_pe_ms) / self.no_pe_ms


def _overhead_color(pct: float) -> str:
    """Green if RoPE is cheap, yellow if noticeable, red if it hurts."""
    if math.isnan(pct):
        return C.GREY
    if pct < 10.0:
        return C.GREEN
    if pct < 30.0:
        return C.YELLOW
    return C.RED


def run_benchmark(cfg: BenchConfig) -> list[Row]:
    torch.manual_seed(cfg.seed)
    rows: list[Row] = []

    header("Part 1 / Latency: how much does RoPE cost?")
    note(
        "For each sequence length we run attention twice on identical random\n"
        "q/k/v tensors: once plain, once with RoPE applied to q and k. We report\n"
        "the median time per forward pass and the extra cost RoPE adds."
    )
    print(
        f"\n{C.DIM}device={cfg.device.type}  dtype={cfg.dtype}  batch={cfg.batch}  "
        f"heads={cfg.heads}  head_dim={cfg.head_dim}  causal={cfg.causal}  "
        f"(warmup={cfg.warmup}, iters={cfg.iters}){C.RESET}\n"
    )

    cols = f"{'seq_len':>8} | {'no-PE (ms)':>12} | {'RoPE (ms)':>12} | {'overhead':>10}"
    print(f"{C.BOLD}{cols}{C.RESET}")
    print(f"{C.GREY}{'-' * len(cols)}{C.RESET}")

    with torch.inference_mode():
        for seq_len in cfg.seq_lens:
            shape = (cfg.batch, cfg.heads, seq_len, cfg.head_dim)
            q = torch.randn(shape, device=cfg.device, dtype=cfg.dtype)
            k = torch.randn(shape, device=cfg.device, dtype=cfg.dtype)
            v = torch.randn(shape, device=cfg.device, dtype=cfg.dtype)
            cos, sin = build_rope_cache(
                seq_len, cfg.head_dim, base=cfg.base, device=cfg.device, dtype=cfg.dtype
            )

            no_pe = time_fn(
                lambda: attention_no_pe(q, k, v, causal=cfg.causal),
                cfg.warmup, cfg.iters, cfg.device,
            )
            rope = time_fn(
                lambda: attention_rope(q, k, v, cos, sin, causal=cfg.causal),
                cfg.warmup, cfg.iters, cfg.device,
            )

            row = Row(seq_len, no_pe * 1e3, rope * 1e3)
            rows.append(row)
            oc = _overhead_color(row.overhead_pct)
            print(
                f"{C.CYAN}{row.seq_len:>8}{C.RESET} | "
                f"{row.no_pe_ms:>12.3f} | "
                f"{row.rope_ms:>12.3f} | "
                f"{oc}{row.overhead_pct:>9.1f}%{C.RESET}"
            )

    if rows:
        avg = sum(r.overhead_pct for r in rows) / len(rows)
        oc = _overhead_color(avg)
        print(
            f"\n{C.BOLD}Takeaway:{C.RESET} RoPE adds {oc}{avg:.1f}%{C.RESET} on average "
            f"here. Its cost is a fixed O(seq * head_dim) elementwise pass, so it\n"
            f"shrinks in relative terms as the O(seq^2) attention matmul dominates."
        )

    return rows


# ===========================================================================
# Behavior demo: the part that explains why anyone bothers with RoPE.
# ===========================================================================
def behavior_demo(cfg: BenchConfig) -> None:
    torch.manual_seed(cfg.seed + 1)
    # The demo wants float64 for tight tolerances, but MPS has no float64 path.
    # This part isn't perf-sensitive, so on MPS we just run it on the CPU.
    if cfg.device.type == "mps":
        device, dtype = torch.device("cpu"), torch.float64
    else:
        device, dtype = cfg.device, torch.float64
    # Threshold for the relative-invariance check. RoPE's rotation accumulates
    # a little float error as we slide the pair, so even in float64 the spread
    # sits around 1e-6 rather than machine epsilon.
    tol = 1e-4 if dtype == torch.float32 else 1e-5
    d = cfg.head_dim
    seq = 64
    cos, sin = build_rope_cache(seq, d, base=cfg.base, device=device, dtype=dtype)

    print()
    header("Part 2 / Behavior: what RoPE actually buys you")
    note(
        "Latency only tells you the price. This half shows the product. We probe\n"
        "the attention score for a single fixed query/key pair, moving it around\n"
        "in the sequence, and watch how the score does (and does not) react."
    )

    # Single fixed query and key content vector, placed at various positions.
    q = torch.randn(d, device=device, dtype=dtype)
    k = torch.randn(d, device=device, dtype=dtype)

    def roped_score(i: int, j: int) -> float:
        qi = apply_rope(q.view(1, 1, 1, d), cos[i : i + 1], sin[i : i + 1]).flatten()
        kj = apply_rope(k.view(1, 1, 1, d), cos[j : j + 1], sin[j : j + 1]).flatten()
        return float(qi @ kj)

    # 1) Relative-position invariance: shifting a (query, key) pair by the same
    #    offset leaves the score unchanged. The score sees only i - j.
    print(f"\n{C.BOLD}{C.BLUE}[1] Relative-position invariance{C.RESET}")
    note(
        "    Keep the content fixed, slide the (query, key) pair right by s.\n"
        "    The score barely moves: RoPE encodes only the gap i - j."
    )
    for (i, j) in [(2, 5), (10, 3)]:
        vals = [roped_score(i + s, j + s) for s in range(6)]
        spread = max(vals) - min(vals)
        ok = C.GREEN if spread < tol else C.YELLOW
        print(
            f"    rel offset i-j = {C.CYAN}{i - j:+d}{C.RESET}:  "
            f"score = {vals[0]:+.5f}   "
            f"max spread over 6 shifts = {ok}{spread:.2e}{C.RESET}"
        )

    # 2) Absolute placement is now visible: different relative offsets differ.
    print(f"\n{C.BOLD}{C.BLUE}[2] Relative position is now visible{C.RESET}")
    note("    Different gaps to the same query give genuinely different scores.")
    for j in range(5):
        print(
            f"    score(query@5, key@{j})  rel={C.CYAN}{5 - j:+d}{C.RESET}:  "
            f"{roped_score(5, j):+.5f}"
        )

    # 3) Without any PE, attention is permutation-equivariant: permute the keys
    #    and the logit columns just permute with them. Order is invisible.
    print(f"\n{C.BOLD}{C.BLUE}[3] Without PE, attention can't see order{C.RESET}")
    note(
        "    Permuting the keys just permutes the logit columns identically.\n"
        "    Raw attention is permutation-equivariant: word order is invisible."
    )
    Q = torch.randn(4, d, device=device, dtype=dtype)
    K = torch.randn(6, d, device=device, dtype=dtype)
    perm = torch.tensor([3, 0, 5, 1, 4, 2], device=device)
    logits = Q @ K.t()
    logits_permuted_keys = Q @ K[perm].t()
    diff = (logits[:, perm] - logits_permuted_keys).abs().max().item()
    ok = C.GREEN if diff < 1e-9 else C.RED
    print(f"    max | permute(logits) - logits(permuted keys) | = {ok}{diff:.2e}{C.RESET}")
    print(
        f"    {C.GREY}=> 'dog bites man' and 'man bites dog' look identical to "
        f"raw attention.{C.RESET}"
    )

    # 4) Sanity check: our hand-written SDPA matches torch's reference kernel.
    print(f"\n{C.BOLD}{C.BLUE}[4] Sanity check{C.RESET}")
    note("    Our hand-written SDPA must match torch's reference kernel.")
    shape = (2, cfg.heads, 32, d)
    qq = torch.randn(shape, device=device, dtype=dtype)
    kk = torch.randn(shape, device=device, dtype=dtype)
    vv = torch.randn(shape, device=device, dtype=dtype)
    mine = scaled_dot_product_attention(qq, kk, vv, causal=True)
    ref = F.scaled_dot_product_attention(qq, kk, vv, is_causal=True)
    err = (mine - ref).abs().max().item()
    ok = C.GREEN if err < 1e-9 else C.RED
    status = "PASS" if err < 1e-9 else "CHECK"
    print(f"    max | mine - reference | = {ok}{err:.2e}  [{status}]{C.RESET}")
    rule("=", color=C.CYAN)


# ===========================================================================
# CLI.
# ===========================================================================
def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def resolve_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark bare-bones attention with and without RoPE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=64, help="must be even")
    p.add_argument("--seq-lens", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    p.add_argument("--causal", action="store_true", help="apply a causal mask")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--base", type=float, default=10000.0, help="RoPE frequency base")
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | mps")
    p.add_argument(
        "--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"]
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", type=str, default=None, help="optional path to write results as CSV")
    p.add_argument("--no-demo", action="store_true", help="skip the behavior demo")
    p.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not _color_enabled(args.no_color):
        C.disable()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    print(f"\n{C.BOLD}{C.MAGENTA}RoPE attention benchmark{C.RESET}")
    note("Comparing scaled dot-product attention with and without Rotary PE.")

    if dtype != torch.float32 and device.type == "cpu":
        print(
            f"{C.YELLOW}Note:{C.RESET} half precision on CPU is slow and noisy; "
            f"float32 is recommended there."
        )

    cfg = BenchConfig(
        batch=args.batch,
        heads=args.heads,
        head_dim=args.head_dim,
        seq_lens=args.seq_lens,
        causal=args.causal,
        warmup=args.warmup,
        iters=args.iters,
        base=args.base,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )

    rows = run_benchmark(cfg)

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["seq_len", "no_pe_ms", "rope_ms", "overhead_pct"])
            for r in rows:
                writer.writerow(
                    [r.seq_len, f"{r.no_pe_ms:.6f}", f"{r.rope_ms:.6f}", f"{r.overhead_pct:.4f}"]
                )
        print(f"\n{C.GREEN}Wrote {len(rows)} rows to {args.csv}{C.RESET}")

    if not args.no_demo:
        behavior_demo(cfg)

    print(f"\n{C.BOLD}{C.GREEN}Done.{C.RESET}\n")


if __name__ == "__main__":
    main()
