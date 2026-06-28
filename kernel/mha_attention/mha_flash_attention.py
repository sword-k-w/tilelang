"""
MHA FlashAttention forward pass.

Layout: BHSD, i.e. Q/K/V/Output are shaped [batch, heads, seq, dim].

This file is intentionally structured as a simple project frame:
  - `mha_flash_attn`: my TileLang implementation.
  - `ref_mha`: non-fused PyTorch reference.
  - `pytorch_sdpa`: PyTorch fused scaled-dot-product attention path.
  - `main`: correctness checks plus benchmark comparison against the official
    TileLang MHA example and PyTorch implementations.

Example:
    python kernel/mha_attention/mha_flash_attention.py
    python kernel/mha_attention/mha_flash_attention.py --seq_q 1024 --seq_kv 1024 --is_causal
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples" / "flash_attention"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from example_mha_fwd_bhsd import flashattn as official_mha_flashattn  # noqa: E402


@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def mha_flash_attn(
    batch: int,
    heads: int,
    seq_q: int,
    seq_kv: int,
    dim: int,
    is_causal: bool = False,
    block_M: int = 64,
    block_N: int = 64,
    num_stages: int = 1,
    threads: int = 128,
):
    # my implementation of mha flash attention forward bhsd kernel (reference: official example_mha_fwd_bhsd)
    """TileLang MHA FlashAttention forward kernel in BHSD layout."""
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    q_shape = [batch, heads, seq_q, dim]
    kv_shape = [batch, heads, seq_kv, dim]
    dtype = T.float16
    accum_dtype = T.float32

    past_len = seq_kv - seq_q
    assert past_len >= 0, "seq_kv must be greater than or equal to seq_q"

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)

            scores = T.alloc_fragment([block_M, block_N], accum_dtype)
            scores_cast = T.alloc_fragment([block_M, block_N], dtype)
            out = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
            T.fill(out, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(T.ceildiv(seq_kv, block_N), T.ceildiv((bx + 1) * block_M + past_len, block_N))
                if is_causal
                else T.ceildiv(seq_kv, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bz, by, k * block_N : (k + 1) * block_N, :], K_shared)

                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        q_idx = bx * block_M + i + past_len
                        k_idx = k * block_N + j
                        scores[i, j] = T.if_then_else(q_idx >= k_idx, 0, -T.infinity(scores.dtype))
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        scores[i, j] = T.if_then_else(k * block_N + j >= seq_kv, -T.infinity(scores.dtype), 0)

                T.gemm(Q_shared, K_shared, scores, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(scores, scores_max, dim=1, clear=False)

                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)

                for i, j in T.Parallel(block_M, dim):
                    out[i, j] *= scores_scale[i]

                for i, j in T.Parallel(block_M, block_N):
                    scores[i, j] = T.exp2(scores[i, j] * scale - scores_max[i] * scale)

                T.reduce_sum(scores, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                T.copy(scores, scores_cast)
                T.copy(V[bz, by, k * block_N : (k + 1) * block_N, :], V_shared)
                T.gemm(scores_cast, V_shared, out, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, dim):
                out[i, j] /= logsum[i]

            T.copy(out, O_shared)
            T.copy(O_shared, Output[bz, by, bx * block_M : (bx + 1) * block_M, :])

    return main


def causal_mask(seq_q: int, seq_kv: int, device: torch.device | str) -> torch.Tensor:
    """Right-aligned causal mask for seq_q <= seq_kv."""
    return torch.tril(
        torch.ones(seq_q, seq_kv, device=device, dtype=torch.bool),
        diagonal=seq_kv - seq_q,
    )


def ref_mha(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
    """Manual PyTorch MHA reference in BHSD layout."""
    dim = Q.size(-1)
    scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / math.sqrt(dim)

    if is_causal:
        mask = causal_mask(Q.size(2), K.size(2), Q.device)
        scores = scores.masked_fill(~mask, float("-inf"))

    attn = F.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", attn, V)


def pytorch_sdpa(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
    """PyTorch scaled-dot-product attention in BHSD layout."""
    if is_causal and Q.size(2) != K.size(2):
        # PyTorch's plain is_causal path is not right-aligned for this decode
        # shape, so provide the same right-aligned mask as the TileLang kernel.
        mask = causal_mask(Q.size(2), K.size(2), Q.device)
        return F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, is_causal=False)
    return F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)


def attention_flops(batch: int, heads: int, seq_q: int, seq_kv: int, dim: int, is_causal: bool) -> float:
    """Approximate forward FLOPs for QK^T and P@V."""
    if is_causal:
        past_len = seq_kv - seq_q
        visible_pairs = seq_q * (past_len + 1) + seq_q * (seq_q - 1) / 2
    else:
        visible_pairs = seq_q * seq_kv
    return 4.0 * batch * heads * visible_pairs * dim


def benchmark_one(name: str, fn, flops: float, warmup: int, rep: int) -> tuple[str, float, float]:
    latency = do_bench(fn, warmup=warmup, rep=rep)
    return name, latency, flops / latency * 1e-9


def print_benchmark_table(rows: list[tuple[str, float, float]]) -> None:
    print("\n--- Performance ---")
    print(f"{'Backend':<32} {'Latency (ms)':<15} {'TFLOPS':<10}")
    print("-" * 60)
    for name, latency, perf in rows:
        print(f"{name:<32} {latency:<15.4f} {perf:<10.2f}")

    mine = rows[0][1]
    print("\nSpeedup relative to my implementation:")
    for name, latency, _ in rows[1:]:
        print(f"  {name:<28} {latency / mine:.2f}x")


def main(
    batch: int = 1,
    heads: int = 16,
    seq_q: int = 512,
    seq_kv: int = 512,
    dim: int = 64,
    is_causal: bool = False,
    block_M: int = 64,
    block_N: int = 64,
    num_stages: int = 1,
    threads: int = 128,
    warmup: int = 25,
    rep: int = 100,
    skip_correctness: bool = False,
    save_source: str | None = None,
):
    dtype = torch.float16
    device = "cuda"
    assert seq_kv >= seq_q, "seq_kv must be greater than or equal to seq_q"

    print("=" * 70)
    print("MHA FlashAttention Forward")
    print("=" * 70)
    print("Layout: BHSD")
    print(f"batch={batch}, heads={heads}, seq_q={seq_q}, seq_kv={seq_kv}, dim={dim}, causal={is_causal}")
    print(f"block_M={block_M}, block_N={block_N}, num_stages={num_stages}, threads={threads}")
    print(f"warmup={warmup}, rep={rep}")

    Q = torch.randn(batch, heads, seq_q, dim, device=device, dtype=dtype)
    K = torch.randn(batch, heads, seq_kv, dim, device=device, dtype=dtype)
    V = torch.randn(batch, heads, seq_kv, dim, device=device, dtype=dtype)

    kernel_mine = mha_flash_attn(
        batch,
        heads,
        seq_q,
        seq_kv,
        dim,
        is_causal,
        block_M=block_M,
        block_N=block_N,
        num_stages=num_stages,
        threads=threads,
    )
    kernel_official = official_mha_flashattn(
        batch,
        heads,
        seq_q,
        seq_kv,
        dim,
        is_causal,
        block_M=block_M,
        block_N=block_N,
        num_stages=num_stages,
        threads=threads,
    )

    if not skip_correctness:
        reference = ref_mha(Q, K, V, is_causal=is_causal)
        mine = kernel_mine(Q, K, V)
        official = kernel_official(Q, K, V)
        sdpa = pytorch_sdpa(Q, K, V, is_causal=is_causal)
        torch.testing.assert_close(mine, reference, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(official, reference, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(sdpa, reference, rtol=1e-2, atol=1e-2)
        print("\n[PASS] My kernel, official kernel, and PyTorch SDPA match manual PyTorch reference.")

    if save_source:
        source_path = Path(save_source)
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(kernel_mine.get_kernel_source())
        print(f"\nCUDA source saved to: {source_path}")

    flops = attention_flops(batch, heads, seq_q, seq_kv, dim, is_causal)
    rows = [
        benchmark_one("My TileLang MHA", lambda: kernel_mine(Q, K, V), flops, warmup, rep),
        benchmark_one("Official TileLang MHA", lambda: kernel_official(Q, K, V), flops, warmup, rep),
        benchmark_one("PyTorch SDPA", lambda: pytorch_sdpa(Q, K, V, is_causal=is_causal), flops, warmup, rep),
        benchmark_one("PyTorch manual", lambda: ref_mha(Q, K, V, is_causal=is_causal), flops, warmup, rep),
    ]
    print_benchmark_table(rows)
    return kernel_mine


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MHA FlashAttention forward benchmark")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--seq_q", type=int, default=512)
    parser.add_argument("--seq_kv", type=int, default=512)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--is_causal", action="store_true", default=False)
    parser.add_argument("--block_M", type=int, default=64)
    parser.add_argument("--block_N", type=int, default=64)
    parser.add_argument("--num_stages", type=int, default=1)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--skip_correctness", action="store_true", default=False)
    parser.add_argument("--save_source", type=str, default=None)
    args = parser.parse_args()

    main(
        batch=args.batch,
        heads=args.heads,
        seq_q=args.seq_q,
        seq_kv=args.seq_kv,
        dim=args.dim,
        is_causal=args.is_causal,
        block_M=args.block_M,
        block_N=args.block_N,
        num_stages=args.num_stages,
        threads=args.threads,
        warmup=args.warmup,
        rep=args.rep,
        skip_correctness=args.skip_correctness,
        save_source=args.save_source,
    )
