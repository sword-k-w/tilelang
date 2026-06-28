"""
GQA (Grouped Query Attention) Flash Attention — Forward Pass
=============================================================
Layout: BSHD — [batch, seq, heads, dim]
KV heads < Q heads, mapped via kv_head = q_head // num_groups.

Reference: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact
Attention with IO-Awareness" (NeurIPS 2022).

To run:
    python kernel/gqa_attention/gqa_flash_attention.py
    python kernel/gqa_attention/gqa_flash_attention.py --seq_len 2048 --is_causal
"""

import sys
import os
import itertools

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune
from tilelang.profiler import do_bench

# Import TileLang's official GQA Flash Attention for comparison
# Resolve relative to THIS file so it works on any machine
_examples_dir = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "flash_attention")
_examples_dir = os.path.abspath(_examples_dir)
if _examples_dir not in sys.path:
    sys.path.insert(0, _examples_dir)
from example_gqa_fwd_bshd import flashattn as official_gqa_flashattn


# =============================================================================
# Autotune Search Space
# =============================================================================
def _get_gqa_configs(
    batch=None,
    heads=None,
    kv_heads=None,
    seq_len=None,
    dim=None,
    is_causal=None,
    block_sizes=(64, 128),
    thread_options=(128, 256),
    num_stages_range=(1, 2, 3),
    max_shared_mem=100 * 1024,
    warp_alignment=16,
    dtype_bytes=2,
):
    """Generate valid (block_M, block_N, num_stages, threads) configs for autotuning.

    Filters by:
      - threads divisible by 32 (warp size)
      - block_M / warp_count and block_N / warp_count alignment for MMA
      - estimated shared memory budget (Q + K + V tiles, with pipeline stages)
    Accepts the kernel's shape args so it can be passed as a callable to @autotune
    and reuse `dim` for SMEM sizing — extra args are ignored.
    """
    assert dim is not None, "_get_gqa_configs requires `dim` to size shared memory"
    valid = []
    for block_M, block_N in itertools.product(block_sizes, repeat=2):
        for threads in thread_options:
            if threads % 32 != 0:
                continue
            warp_count = threads // 32
            if block_M % warp_count or block_N % warp_count:
                continue
            warp_M = block_M // warp_count
            warp_N = block_N // warp_count
            if warp_M % warp_alignment != 0 or warp_N % warp_alignment != 0:
                continue
            # Rough SMEM bound: Q + K + V tiles (mirrors the official example's heuristic).
            shared_mem = 2 * dtype_bytes * dim * (block_M + block_N)
            if shared_mem > max_shared_mem:
                continue
            for num_stages in num_stages_range:
                valid.append({
                    "block_M": block_M,
                    "block_N": block_N,
                    "num_stages": num_stages,
                    "threads": threads,
                })
    assert valid, "No valid autotune configs were produced — relax the search space."
    return valid


# =============================================================================
# GQA Flash Attention Kernel — prim_func builder
# =============================================================================
def _build_gqa_prim_func(
    batch: int,
    heads: int,
    kv_heads: int,
    seq_len: int,
    dim: int,
    is_causal: bool,
    block_M: int,
    block_N: int,
    num_stages: int,
    threads: int,
):
    """Construct and return the TileLang prim_func for GQA Flash Attention.

    Shared by the direct-compile entry point `gqa_flash_attn` and the autotuned
    entry point `gqa_flash_attn_tuned`.
    """
    # ---- Validate ----
    assert heads % kv_heads == 0, f"heads ({heads}) must be divisible by kv_heads ({kv_heads})"

    num_groups = heads // kv_heads

    # scale = 1/sqrt(dim) * log2(e) — use exp2/log2 for hardware efficiency
    scale = (1.0 / dim) ** 0.5 * 1.44269504

    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, kv_heads, dim]
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        # 3D kernel grid:
        #   bx → Q row block    (which block_M rows of the sequence)
        #   by → attention head  (0 .. heads-1)
        #   bz → batch element   (which sequence)
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
            # ---- Shared Memory (on-chip SRAM) ----
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)

            # ---- Register Fragments ----
            S_local = T.alloc_fragment([block_M, block_N], accum_dtype)  # QK^T scores (fp32)
            S_local_fp16 = T.alloc_fragment([block_M, block_N], dtype)  # scores cast to fp16
            O_local = T.alloc_fragment([block_M, dim], accum_dtype)  # output accumulator (fp32)
            S_max = T.alloc_fragment([block_M], accum_dtype)  # running per-row max
            S_max_prev = T.alloc_fragment([block_M], accum_dtype)  # previous per-row max
            rescale = T.alloc_fragment([block_M], accum_dtype)  # rescale factor
            S_exp_sum = T.alloc_fragment([block_M], accum_dtype)  # per-row softmax sum
            total_sum = T.alloc_fragment([block_M], accum_dtype)  # running denominator

            # ---- GQA: map Q head to KV head ----
            # Multiple Q heads share one KV head. For example, heads=32, kv_heads=8:
            #   Q head 0..3  →  KV head 0
            #   Q head 4..7  →  KV head 1
            #   ...
            kv_head = by // num_groups

            # ---- Step 1: Load Q block into shared memory (loaded ONCE) ----
            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)

            # ---- Step 2: Initialize running state ----
            T.fill(O_local, 0)
            T.fill(total_sum, 0)
            T.fill(S_max, -T.infinity(accum_dtype))

            # ---- Step 3: Determine KV loop range ----
            # With causal mask, later Q blocks can see more KV blocks.
            # The last row of this Q block is at position: (bx + 1) * block_M - 1
            # KV blocks beyond that position contain only future tokens → skip.
            if is_causal:
                loop_range = T.min(
                    T.ceildiv(seq_len, block_N),
                    T.ceildiv((bx + 1) * block_M, block_N),
                )
            else:
                loop_range = T.ceildiv(seq_len, block_N)

            # ---- Step 4: Main loop — iterate over KV blocks ----
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                # ---- 4a. Load K block into shared memory ----
                # NOTE: K uses kv_head (not by) because KV has fewer heads
                T.copy(K[bz, k * block_N : (k + 1) * block_N, kv_head, :], K_shared)

                # ---- 4b. Apply mask before gemm (S_local is ACCUMULATED into) ----
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        S_local[i, j] = T.if_then_else(
                            bx * block_M + i >= k * block_N + j,
                            0,
                            -T.infinity(S_local.dtype),
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        S_local[i, j] = T.if_then_else(
                            k * block_N + j >= seq_len,
                            -T.infinity(S_local.dtype),
                            0,
                        )
                # ---- 4c. Compute S = Q @ K^T ----
                T.gemm(Q_shared, K_shared, S_local, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # ---- 4d. Online softmax rescaling (Dao et al. 2022, Algorithm 1) ----
                # (1) Save old per-row max
                T.copy(S_max, S_max_prev)
                T.fill(S_max, -T.infinity(accum_dtype))

                # (2–3) Row-wise max of new S block; update running max
                T.reduce_max(S_local, S_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    S_max[i] = T.max(S_max[i], S_max_prev[i])
                    # (4) Rescale factor: exp((m_old - m_new) / sqrt(d))
                    rescale[i] = T.exp2(S_max_prev[i] * scale - S_max[i] * scale)

                # (5) Rescale old output accumulation
                for i, j in T.Parallel(block_M, dim):
                    O_local[i, j] *= rescale[i]

                # (6) P = softmax(S) = exp((S - m_new) / sqrt(d))
                for i, j in T.Parallel(block_M, block_N):
                    S_local[i, j] = T.exp2(S_local[i, j] * scale - S_max[i] * scale)

                # (7) Update running softmax denominator
                T.reduce_sum(S_local, S_exp_sum, dim=1)
                for i in T.Parallel(block_M):
                    total_sum[i] = total_sum[i] * rescale[i] + S_exp_sum[i]

                # (8) Cast P to fp16 for tensor core gemm
                T.copy(S_local, S_local_fp16)

                # ---- 4e. Load V, accumulate P @ V → O_local ----
                T.copy(V[bz, k * block_N : (k + 1) * block_N, kv_head, :], V_shared)
                T.gemm(S_local_fp16, V_shared, O_local, policy=T.GemmWarpPolicy.FullRow)

            # ---- Step 5: Final normalization O /= sum(P) ----
            for i, j in T.Parallel(block_M, dim):
                O_local[i, j] /= total_sum[i]

            # ---- Step 6: Write output back to HBM ----
            # Route through O_shared for coalesced memory writes.
            T.copy(O_local, O_shared)
            T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return main


# =============================================================================
# Public entry points
# =============================================================================
@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def gqa_flash_attn(
    batch: int,
    heads: int,
    kv_heads: int,
    seq_len: int,
    dim: int,
    is_causal: bool = False,
    block_M: int = 64,
    block_N: int = 64,
    num_stages: int = 2,
    threads: int = 128,
):
    """Direct-compile GQA Flash Attention forward (no autotuning)."""
    return _build_gqa_prim_func(
        batch, heads, kv_heads, seq_len, dim, is_causal,
        block_M, block_N, num_stages, threads,
    )


@autotune(configs=_get_gqa_configs, warmup=10, rep=10)
@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def gqa_flash_attn_tuned(
    batch: int,
    heads: int,
    kv_heads: int,
    seq_len: int,
    dim: int,
    is_causal: bool = False,
    block_M: int = 64,
    block_N: int = 64,
    num_stages: int = 2,
    threads: int = 128,
):
    """Autotuned GQA Flash Attention forward — searches (block_M, block_N, num_stages, threads)."""
    return _build_gqa_prim_func(
        batch, heads, kv_heads, seq_len, dim, is_causal,
        block_M, block_N, num_stages, threads,
    )


# =============================================================================
# GQA Group-Aware Kernel — KV shared across the num_groups Q heads of one group
# =============================================================================
# Key idea: in plain GQA the grid is (seq_q/block_M, heads, batch) and each
# block loads K/V for its kv_head = by // num_groups. Across the `num_groups`
# Q heads that share a kv_head, K/V is loaded `num_groups` times from HBM.
#
# This variant uses grid (seq_q/block_M, kv_heads, batch). One block handles
# ALL `num_groups` Q heads sharing a kv_head, so K/V is loaded ONCE per (seq_q
# block, kv_head). Expected HBM traffic for K/V drops by `num_groups`x
# (4x for heads=32, kv_heads=8).
#
# Implementation: Q tile grows from [block_M, dim] to [num_groups * block_M, dim]
# packed group-major (rows g*block_M + s ← Q[..., bx*block_M + s, kv_head*ng + g, :]).
# All FA fragments (S, O, max, sum) grow in their row dim by `num_groups`.
# Softmax bookkeeping remains row-wise — rows of different groups are independent.
# Trade-off: register pressure scales with num_groups; use smaller block_M to
# compensate (autotune handles this).
# =============================================================================
def _get_gqa_group_aware_configs(
    batch=None,
    heads=None,
    kv_heads=None,
    seq_len=None,
    dim=None,
    is_causal=None,
    block_M_options=(16, 32, 64),
    block_N_options=(32, 64, 128),
    thread_options=(128, 256),
    num_stages_range=(1, 2, 3),
    max_shared_mem=100 * 1024,
    warp_alignment=16,
    dtype_bytes=2,
):
    """Config search space for the group-aware kernel.

    Differences from `_get_gqa_configs`:
      - block_M is per-Q-position (the kernel internally uses M_tile = num_groups * block_M).
      - Warp alignment check uses M_tile, not block_M.
      - SMEM budget accounts for the larger Q + O tiles (size num_groups * block_M * dim).
    """
    assert dim is not None and kv_heads is not None and heads is not None, (
        "_get_gqa_group_aware_configs requires `dim`, `heads`, `kv_heads`"
    )
    num_groups = heads // kv_heads
    valid = []
    for block_M in block_M_options:
        M_tile = num_groups * block_M
        for block_N in block_N_options:
            for threads in thread_options:
                if threads % 32 != 0:
                    continue
                warp_count = threads // 32
                if M_tile % warp_count or block_N % warp_count:
                    continue
                warp_M = M_tile // warp_count
                warp_N = block_N // warp_count
                if warp_M % warp_alignment != 0 or warp_N % warp_alignment != 0:
                    continue
                # SMEM: Q + K + V + O tiles.
                # Q and O are M_tile * dim each, K and V are block_N * dim each.
                shared_mem = dtype_bytes * dim * (2 * M_tile + 2 * block_N)
                if shared_mem > max_shared_mem:
                    continue
                for num_stages in num_stages_range:
                    valid.append({
                        "block_M": block_M,
                        "block_N": block_N,
                        "num_stages": num_stages,
                        "threads": threads,
                    })
    assert valid, (
        f"No valid group-aware configs for num_groups={num_groups}, dim={dim}. "
        "Try relaxing block_M_options or max_shared_mem."
    )
    return valid


def _build_gqa_group_aware_prim_func(
    batch: int,
    heads: int,
    kv_heads: int,
    seq_len: int,
    dim: int,
    is_causal: bool,
    block_M: int,
    block_N: int,
    num_stages: int,
    threads: int,
):
    """Construct the prim_func for the group-aware GQA kernel.

    `block_M` is per-Q-position. The internal tile spans num_groups * block_M rows.
    """
    assert heads % kv_heads == 0, f"heads ({heads}) must be divisible by kv_heads ({kv_heads})"
    num_groups = heads // kv_heads
    M_tile = num_groups * block_M

    scale = (1.0 / dim) ** 0.5 * 1.44269504

    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, kv_heads, dim]
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        # Grid Y is now `kv_heads`, not `heads` — one block per (seq_q block, kv_head, batch).
        with T.Kernel(T.ceildiv(seq_len, block_M), kv_heads, batch, threads=threads) as (bx, by, bz):
            kv_head = by

            Q_shared = T.alloc_shared([M_tile, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([M_tile, dim], dtype)

            S_local = T.alloc_fragment([M_tile, block_N], accum_dtype)
            S_local_fp16 = T.alloc_fragment([M_tile, block_N], dtype)
            O_local = T.alloc_fragment([M_tile, dim], accum_dtype)
            S_max = T.alloc_fragment([M_tile], accum_dtype)
            S_max_prev = T.alloc_fragment([M_tile], accum_dtype)
            rescale = T.alloc_fragment([M_tile], accum_dtype)
            S_exp_sum = T.alloc_fragment([M_tile], accum_dtype)
            total_sum = T.alloc_fragment([M_tile], accum_dtype)

            # ---- Load Q for all num_groups Q heads sharing this kv_head ----
            # BSHD source slice has shape [block_M, num_groups, dim]; we pack it
            # group-major into Q_shared: row g*block_M + s ← Q[bz, bx*block_M+s, kv_head*ng+g, :]
            for s, g, d in T.Parallel(block_M, num_groups, dim):
                Q_shared[g * block_M + s, d] = Q[
                    bz, bx * block_M + s, kv_head * num_groups + g, d
                ]

            T.fill(O_local, 0)
            T.fill(total_sum, 0)
            T.fill(S_max, -T.infinity(accum_dtype))

            # Causal: same Q-position constraint applies to all groups → unchanged.
            if is_causal:
                loop_range = T.min(
                    T.ceildiv(seq_len, block_N),
                    T.ceildiv((bx + 1) * block_M, block_N),
                )
            else:
                loop_range = T.ceildiv(seq_len, block_N)

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                # ONE K load serves all num_groups Q heads — the GQA bandwidth win.
                T.copy(K[bz, k * block_N : (k + 1) * block_N, kv_head, :], K_shared)

                if is_causal:
                    # Row i = g*block_M + s; q_idx depends only on s (same across groups).
                    for i, j in T.Parallel(M_tile, block_N):
                        s_in_block = i % block_M
                        q_idx = bx * block_M + s_in_block
                        k_idx = k * block_N + j
                        S_local[i, j] = T.if_then_else(
                            q_idx >= k_idx, 0, -T.infinity(S_local.dtype)
                        )
                else:
                    for i, j in T.Parallel(M_tile, block_N):
                        S_local[i, j] = T.if_then_else(
                            k * block_N + j >= seq_len,
                            -T.infinity(S_local.dtype),
                            0,
                        )

                T.gemm(
                    Q_shared, K_shared, S_local,
                    transpose_B=True, policy=T.GemmWarpPolicy.FullRow,
                )

                # Online softmax — identical to single-head FA, just bigger M.
                T.copy(S_max, S_max_prev)
                T.fill(S_max, -T.infinity(accum_dtype))
                T.reduce_max(S_local, S_max, dim=1, clear=False)
                for i in T.Parallel(M_tile):
                    S_max[i] = T.max(S_max[i], S_max_prev[i])
                    rescale[i] = T.exp2(S_max_prev[i] * scale - S_max[i] * scale)

                for i, j in T.Parallel(M_tile, dim):
                    O_local[i, j] *= rescale[i]

                for i, j in T.Parallel(M_tile, block_N):
                    S_local[i, j] = T.exp2(S_local[i, j] * scale - S_max[i] * scale)

                T.reduce_sum(S_local, S_exp_sum, dim=1)
                for i in T.Parallel(M_tile):
                    total_sum[i] = total_sum[i] * rescale[i] + S_exp_sum[i]

                T.copy(S_local, S_local_fp16)

                # ONE V load also serves all num_groups Q heads.
                T.copy(V[bz, k * block_N : (k + 1) * block_N, kv_head, :], V_shared)
                T.gemm(
                    S_local_fp16, V_shared, O_local,
                    policy=T.GemmWarpPolicy.FullRow,
                )

            for i, j in T.Parallel(M_tile, dim):
                O_local[i, j] /= total_sum[i]

            T.copy(O_local, O_shared)
            # Unpack rows back to BSHD heads.
            for s, g, d in T.Parallel(block_M, num_groups, dim):
                Output[bz, bx * block_M + s, kv_head * num_groups + g, d] = (
                    O_shared[g * block_M + s, d]
                )

    return main


@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def gqa_flash_attn_group_aware(
    batch: int,
    heads: int,
    kv_heads: int,
    seq_len: int,
    dim: int,
    is_causal: bool = False,
    block_M: int = 32,
    block_N: int = 64,
    num_stages: int = 2,
    threads: int = 128,
):
    """Direct-compile group-aware GQA Flash Attention (KV shared across groups)."""
    return _build_gqa_group_aware_prim_func(
        batch, heads, kv_heads, seq_len, dim, is_causal,
        block_M, block_N, num_stages, threads,
    )


@autotune(configs=_get_gqa_group_aware_configs, warmup=10, rep=10)
@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def gqa_flash_attn_group_aware_tuned(
    batch: int,
    heads: int,
    kv_heads: int,
    seq_len: int,
    dim: int,
    is_causal: bool = False,
    block_M: int = 32,
    block_N: int = 64,
    num_stages: int = 2,
    threads: int = 128,
):
    """Autotuned group-aware GQA Flash Attention."""
    return _build_gqa_group_aware_prim_func(
        batch, heads, kv_heads, seq_len, dim, is_causal,
        block_M, block_N, num_stages, threads,
    )


# =============================================================================
# Reference Implementation (PyTorch)
# =============================================================================
def ref_gqa(Q, K, V, is_causal: bool = False):
    """
    Reference GQA attention using PyTorch.

    Q: [batch, seq_len, heads, dim]        — BSHD
    K: [batch, seq_len, kv_heads, dim]     — BSHD
    V: [batch, seq_len, kv_heads, dim]     — BSHD
    """
    batch, seq_len, heads, dim = Q.shape
    kv_heads = K.shape[2]
    num_groups = heads // kv_heads

    assert heads % kv_heads == 0
    assert K.shape[2] == kv_heads
    assert V.shape[2] == kv_heads

    # Expand KV to match Q heads: repeat each KV head num_groups times
    # [B, S, HK, D] → [B, S, HQ, D]
    K_expanded = K.repeat_interleave(num_groups, dim=2)
    V_expanded = V.repeat_interleave(num_groups, dim=2)

    # Reshape to BHSD for einsum
    Q_bhsd = Q.permute(0, 2, 1, 3)  # [B, HQ, S, D]
    K_bhsd = K_expanded.permute(0, 2, 1, 3)
    V_bhsd = V_expanded.permute(0, 2, 1, 3)

    # Standard scaled dot-product attention (without flash fusion)
    scores = torch.einsum("bhqd,bhkd->bhqk", Q_bhsd, K_bhsd)
    scores = scores / (dim**0.5)

    if is_causal:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    output_bhsd = torch.einsum("bhqk,bhkd->bhqd", attn_weights, V_bhsd)

    # Convert back to BSHD
    return output_bhsd.permute(0, 2, 1, 3)


# =============================================================================
# Main: Correctness + Performance
# =============================================================================
def main(
    batch: int = 1,
    heads: int = 32,
    kv_heads: int = 8,
    seq_len: int = 1024,
    dim: int = 64,
    is_causal: bool = False,
    block_M: int = 64,
    block_N: int = 64,
    num_stages: int = 2,
    threads: int = 128,
    tune: bool = False,
):
    dtype = torch.float16
    device = "cuda"

    print("=" * 60)
    print("GQA Flash Attention — Forward Pass")
    print("=" * 60)
    print(f"Layout: BSHD  |  batch={batch}, heads={heads}, kv_heads={kv_heads}")
    print(f"seq_len={seq_len}, dim={dim}, causal={is_causal}")
    if tune:
        print("Mode: AUTOTUNE (block_M / block_N / num_stages / threads searched)")
    else:
        print(f"block_M={block_M}, block_N={block_N}, num_stages={num_stages}, threads={threads}")

    # ---- Input data ----
    Q = torch.randn(batch, seq_len, heads, dim, device=device, dtype=dtype)
    K = torch.randn(batch, seq_len, kv_heads, dim, device=device, dtype=dtype)
    V = torch.randn(batch, seq_len, kv_heads, dim, device=device, dtype=dtype)

    # ---- Compile TileLang kernel (tuned or direct) ----
    # When --tune is on, both my kernel AND the official baseline are autotuned
    # (the official's @autotune fires only when tile kwargs are NOT passed).
    # When --tune is off, both run at the user-CLI defaults for an apples-to-apples
    # comparison on the same config.
    official_block_M, official_block_N = block_M, block_N
    official_num_stages, official_threads = num_stages, threads

    if tune:
        # Autotuner sweeps the config space and returns the best JITKernel.
        # `.config`, `.latency`, `.ref_latency` are attached by the tuner.
        kernel = gqa_flash_attn_tuned(
            batch=batch,
            heads=heads,
            kv_heads=kv_heads,
            seq_len=seq_len,
            dim=dim,
            is_causal=is_causal,
        )
        best_cfg = getattr(kernel, "config", None)
        best_lat = getattr(kernel, "latency", None)
        if best_cfg is not None:
            print(f"\n[AUTOTUNE] My best config: {best_cfg}")
        if best_lat is not None:
            print(f"[AUTOTUNE] My best latency during search: {best_lat:.4f} ms")
    else:
        kernel = gqa_flash_attn(
            batch=batch,
            heads=heads,
            kv_heads=kv_heads,
            seq_len=seq_len,
            dim=dim,
            is_causal=is_causal,
            block_M=block_M,
            block_N=block_N,
            num_stages=num_stages,
            threads=threads,
        )

    # ---- Compile group-aware variant (KV shared across num_groups Q heads) ----
    if tune:
        kernel_ga = gqa_flash_attn_group_aware_tuned(
            batch=batch, heads=heads, kv_heads=kv_heads,
            seq_len=seq_len, dim=dim, is_causal=is_causal,
        )
        ga_cfg = getattr(kernel_ga, "config", None)
        if ga_cfg is not None:
            print(f"[AUTOTUNE] Group-aware best config: {ga_cfg}")
    else:
        # Use a smaller default block_M to keep M_tile = num_groups*block_M reasonable.
        kernel_ga = gqa_flash_attn_group_aware(
            batch=batch, heads=heads, kv_heads=kv_heads,
            seq_len=seq_len, dim=dim, is_causal=is_causal,
            block_M=32, block_N=64, num_stages=2, threads=128,
        )

    # ---- Correctness ----
    result = kernel(Q, K, V)
    result_ga = kernel_ga(Q, K, V)
    reference = ref_gqa(Q, K, V, is_causal=is_causal)

    try:
        torch.testing.assert_close(result, reference, rtol=1e-2, atol=1e-2)
        print("\n[PASS] Standard kernel matches PyTorch reference (rtol=1e-2)")
    except AssertionError as e:
        max_err = (result.float() - reference.float()).abs().max().item()
        rel_err = ((result.float() - reference.float()).abs() / (reference.float().abs() + 1e-8)).max().item()
        print(f"\n[FAIL] Standard mismatch!  Max abs err: {max_err:.6f}, Max rel err: {rel_err:.6f}")
        print(e)

    try:
        torch.testing.assert_close(result_ga, reference, rtol=1e-2, atol=1e-2)
        print("[PASS] Group-aware kernel matches PyTorch reference (rtol=1e-2)")
    except AssertionError as e:
        max_err = (result_ga.float() - reference.float()).abs().max().item()
        rel_err = ((result_ga.float() - reference.float()).abs() / (reference.float().abs() + 1e-8)).max().item()
        print(f"[FAIL] Group-aware mismatch!  Max abs err: {max_err:.6f}, Max rel err: {rel_err:.6f}")
        print(e)

    # ---- Generated CUDA source ----
    src_path = f"/tmp/gqa_flash_attn_s{seq_len}_d{dim}.cu"
    with open(src_path, "w") as f:
        f.write(kernel.get_kernel_source())
    print(f"\nCUDA source saved to: {src_path}")

    # ---- Performance ----
    num_groups = heads // kv_heads

    # Compile TileLang's official GQA Flash Attention.
    # The official API uses `groups` (Q heads per KV head) instead of `kv_heads`.
    # When `tune=True`, we OMIT tile-size kwargs so the official's own @autotune
    # decorator actually searches — otherwise providing all tunables triggers the
    # "Tunable parameters already provided ... Skipping compilation" path and
    # silently disables its autotune (which would make the comparison unfair).
    if tune:
        print("\n[BENCH] Letting official kernel autotune itself (no tile kwargs)...")
        kernel_official = official_gqa_flashattn(
            batch, heads, seq_len, dim, is_causal,
            groups=num_groups,
        )
        off_cfg = getattr(kernel_official, "config", None)
        if off_cfg is not None:
            print(f"[BENCH] Official tuned config: {off_cfg}")
    else:
        kernel_official = official_gqa_flashattn(
            batch, heads, seq_len, dim, is_causal,
            groups=num_groups,
            block_M=official_block_M,
            block_N=official_block_N,
            num_stages=official_num_stages,
            threads=official_threads,
        )

    # ---- CUDA source diff (mine vs official) ----
    # If both kernels lower to byte-identical CUDA, any perf gap is measurement noise.
    # If they differ, the diff localizes which lowering choice changed.
    off_src_path = f"/tmp/gqa_flash_attn_official_s{seq_len}_d{dim}.cu"
    with open(off_src_path, "w") as f:
        f.write(kernel_official.get_kernel_source())
    print(f"Official CUDA source saved to: {off_src_path}")

    import difflib
    mine_lines = kernel.get_kernel_source().splitlines(keepends=True)
    off_lines = kernel_official.get_kernel_source().splitlines(keepends=True)
    if mine_lines == off_lines:
        print(f"\n[CUDA DIFF] Identical ({len(mine_lines)} lines each).")
        print("            → any latency gap is measurement noise, not a code difference.")
    else:
        # Summary stats
        added = sum(1 for l in difflib.unified_diff(off_lines, mine_lines, n=0) if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in difflib.unified_diff(off_lines, mine_lines, n=0) if l.startswith("-") and not l.startswith("---"))
        print(f"\n[CUDA DIFF] mine={len(mine_lines)} lines  official={len(off_lines)} lines  "
              f"(+{added} / -{removed} vs official)")
        # Write full unified diff to disk for inspection
        diff_path = f"/tmp/gqa_flash_attn_diff_s{seq_len}_d{dim}.diff"
        with open(diff_path, "w") as f:
            f.writelines(difflib.unified_diff(
                off_lines, mine_lines,
                fromfile="official.cu", tofile="mine.cu", n=3,
            ))
        print(f"            full diff written to: {diff_path}")
        # Print the first ~40 lines of the diff inline so the user sees the shape immediately
        print("\n            --- first hunks (head -40 of diff) ---")
        head = list(difflib.unified_diff(
            off_lines, mine_lines,
            fromfile="official.cu", tofile="mine.cu", n=3,
        ))[:40]
        for line in head:
            print("            " + line.rstrip())

    # PyTorch sdpa (native flash attention via cuDNN)
    def ptx_sdpa(q, k, v, causal):
        # F.scaled_dot_product_attention expects BHSD layout
        q_bhsd = q.permute(0, 2, 1, 3)
        k_bhsd = k.permute(0, 2, 1, 3)
        v_bhsd = v.permute(0, 2, 1, 3)
        # Expand KV for GQA
        num_g = q.shape[2] // k.shape[2]
        k_exp = k_bhsd.repeat_interleave(num_g, dim=1)
        v_exp = v_bhsd.repeat_interleave(num_g, dim=1)
        out = F.scaled_dot_product_attention(q_bhsd, k_exp, v_exp, is_causal=causal)
        return out.permute(0, 2, 1, 3)

    # PyTorch manual (no fusion — writes full [seq, seq] to HBM)
    def ptx_manual(q, k, v, causal):
        return ref_gqa(q, k, v, is_causal=causal)

    lat_mine = do_bench(lambda: kernel(Q, K, V), warmup=500, rep=100)
    lat_mine_ga = do_bench(lambda: kernel_ga(Q, K, V), warmup=500, rep=100)
    lat_official = do_bench(lambda: kernel_official(Q, K, V), warmup=500, rep=100)
    lat_sdpa = do_bench(lambda: ptx_sdpa(Q, K, V, is_causal), warmup=500, rep=100)
    lat_manual = do_bench(lambda: ptx_manual(Q, K, V, is_causal), warmup=500, rep=100)

    flops = 2.0 * batch * heads * seq_len * seq_len * dim * 2  # 2 matmuls (QK^T + PV)
    if is_causal:
        flops *= 0.5

    print("\n--- Performance ---")
    print(f"{'Backend':<35} {'Latency (ms)':<15} {'TFlops':<10}")
    print("-" * 60)
    print(f"{'My GQA Flash Attn (standard)':<35} {lat_mine:<15.4f} {flops / lat_mine * 1e-9:<10.2f}")
    print(f"{'My GQA Flash Attn (group-aware)':<35} {lat_mine_ga:<15.4f} {flops / lat_mine_ga * 1e-9:<10.2f}")
    print(f"{'TileLang Official GQA':<35} {lat_official:<15.4f} {flops / lat_official * 1e-9:<10.2f}")
    print(f"{'PyTorch sdpa (cuDNN FA)':<35} {lat_sdpa:<15.4f} {flops / lat_sdpa * 1e-9:<10.2f}")
    print(f"{'PyTorch manual (no fusion)':<35} {lat_manual:<15.4f} {flops / lat_manual * 1e-9:<10.2f}")
    print(f"\nStandard      vs Official: {lat_official / lat_mine:.2f}x")
    print(f"Group-aware   vs Official: {lat_official / lat_mine_ga:.2f}x")
    print(f"Group-aware   vs Standard: {lat_mine / lat_mine_ga:.2f}x")

    return kernel


# =============================================================================
# Sweep Benchmark — seq_len vs latency across backends
# =============================================================================
def benchmark_sweep(
    seq_lens=(512, 1024, 2048, 4096),
    batch: int = 1,
    heads: int = 32,
    kv_heads: int = 8,
    dim: int = 64,
    is_causal: bool = False,
    block_M: int = 64,
    block_N: int = 64,
    num_stages: int = 2,
    threads: int = 128,
    tune: bool = False,
):
    """Run correctness + performance across multiple sequence lengths.

    When `tune=True`, autotune is run per seq_len and the best (block_M, block_N,
    num_stages, threads) are also used for the official baseline so the comparison
    stays apples-to-apples on tile sizes (the gap then measures kernel-level changes).
    """
    dtype = torch.float16
    device = "cuda"
    num_groups = heads // kv_heads

    results = []
    print("=" * 80)
    print(f"GQA Flash Attention — Sweep: seq_len ∈ {list(seq_lens)}")
    print(f"batch={batch}, heads={heads}, kv_heads={kv_heads}, dim={dim}, causal={is_causal}")
    if tune:
        print("Mode: AUTOTUNE per seq_len")
    print("=" * 80)

    for seq_len in seq_lens:
        print(f"\n--- seq_len = {seq_len} ---")

        Q = torch.randn(batch, seq_len, heads, dim, device=device, dtype=dtype)
        K = torch.randn(batch, seq_len, kv_heads, dim, device=device, dtype=dtype)
        V = torch.randn(batch, seq_len, kv_heads, dim, device=device, dtype=dtype)

        # My kernel — autotuned or fixed
        if tune:
            kernel = gqa_flash_attn_tuned(
                batch=batch, heads=heads, kv_heads=kv_heads,
                seq_len=seq_len, dim=dim, is_causal=is_causal,
            )
            cfg = getattr(kernel, "config", None) or {}
            bM = cfg.get("block_M", block_M)
            bN = cfg.get("block_N", block_N)
            ns = cfg.get("num_stages", num_stages)
            th = cfg.get("threads", threads)
            print(f"  [AUTOTUNE] best cfg: block_M={bM}, block_N={bN}, num_stages={ns}, threads={th}")
        else:
            bM, bN, ns, th = block_M, block_N, num_stages, threads
            kernel = gqa_flash_attn(
                batch=batch, heads=heads, kv_heads=kv_heads,
                seq_len=seq_len, dim=dim, is_causal=is_causal,
                block_M=bM, block_N=bN,
                num_stages=ns, threads=th,
            )

        # Official baseline.
        # When tune=True: omit tile kwargs so the official's @autotune actually fires.
        # When tune=False: pass the user-CLI defaults so both sit at the same config.
        if tune:
            kernel_official = official_gqa_flashattn(
                batch, heads, seq_len, dim, is_causal,
                groups=num_groups,
            )
            off_cfg = getattr(kernel_official, "config", None)
            if off_cfg is not None:
                print(f"  [BENCH]    official tuned cfg: block_M={off_cfg.get('block_M')}, "
                      f"block_N={off_cfg.get('block_N')}, num_stages={off_cfg.get('num_stages')}, "
                      f"threads={off_cfg.get('threads')}")
        else:
            kernel_official = official_gqa_flashattn(
                batch, heads, seq_len, dim, is_causal,
                groups=num_groups,
                block_M=block_M, block_N=block_N,
                num_stages=num_stages, threads=threads,
            )

        # Group-aware variant (KV shared across num_groups Q heads)
        if tune:
            kernel_ga = gqa_flash_attn_group_aware_tuned(
                batch=batch, heads=heads, kv_heads=kv_heads,
                seq_len=seq_len, dim=dim, is_causal=is_causal,
            )
            ga_cfg = getattr(kernel_ga, "config", None) or {}
            print(f"  [AUTOTUNE] group-aware cfg: block_M={ga_cfg.get('block_M')}, "
                  f"block_N={ga_cfg.get('block_N')}, num_stages={ga_cfg.get('num_stages')}, "
                  f"threads={ga_cfg.get('threads')}")
        else:
            kernel_ga = gqa_flash_attn_group_aware(
                batch=batch, heads=heads, kv_heads=kv_heads,
                seq_len=seq_len, dim=dim, is_causal=is_causal,
                block_M=32, block_N=64, num_stages=2, threads=128,
            )

        # Correctness
        result = kernel(Q, K, V)
        result_ga = kernel_ga(Q, K, V)
        reference = ref_gqa(Q, K, V, is_causal=is_causal)
        try:
            torch.testing.assert_close(result, reference, rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(result_ga, reference, rtol=1e-2, atol=1e-2)
            print("  [PASS] standard + group-aware correctness")
        except AssertionError as e:
            max_err = (result.float() - reference.float()).abs().max().item()
            max_err_ga = (result_ga.float() - reference.float()).abs().max().item()
            print(f"  [FAIL] max err (std)={max_err:.6f}  max err (ga)={max_err_ga:.6f}")
            print(e)
            continue

        # Benchmarks
        def ptx_sdpa(q, k, v):
            q_b = q.permute(0, 2, 1, 3)
            k_b = k.permute(0, 2, 1, 3)
            v_b = v.permute(0, 2, 1, 3)
            ng = q.shape[2] // k.shape[2]
            k_be = k_b.repeat_interleave(ng, dim=1)
            v_be = v_b.repeat_interleave(ng, dim=1)
            o = F.scaled_dot_product_attention(q_b, k_be, v_be, is_causal=is_causal)
            return o.permute(0, 2, 1, 3)

        torch.cuda.synchronize()
        lat_mine = do_bench(lambda: kernel(Q, K, V), warmup=500, rep=100)
        lat_mine_ga = do_bench(lambda: kernel_ga(Q, K, V), warmup=500, rep=100)
        lat_official = do_bench(lambda: kernel_official(Q, K, V), warmup=500, rep=100)
        lat_sdpa = do_bench(lambda: ptx_sdpa(Q, K, V), warmup=500, rep=100)
        lat_manual = do_bench(lambda: ref_gqa(Q, K, V, is_causal=is_causal), warmup=500, rep=100)

        flops = 4.0 * batch * heads * seq_len * seq_len * dim  # QK^T + PV
        if is_causal:
            flops *= 0.5

        results.append({
            "seq_len": seq_len,
            "mine_ms": lat_mine,
            "mine_ga_ms": lat_mine_ga,
            "mine_tflops": flops / lat_mine * 1e-9,
            "mine_ga_tflops": flops / lat_mine_ga * 1e-9,
            "official_ms": lat_official,
            "sdpa_ms": lat_sdpa,
            "manual_ms": lat_manual,
            "config": {"block_M": bM, "block_N": bN, "num_stages": ns, "threads": th},
        })

        print(f"  Standard: {lat_mine:.4f} ms  |  Group-aware: {lat_mine_ga:.4f} ms  |  "
              f"Official: {lat_official:.4f} ms  |  sdpa: {lat_sdpa:.4f} ms  |  Manual: {lat_manual:.4f} ms")

    # Print summary table
    print("\n" + "=" * 80)
    print("Summary Table")
    print("=" * 80)
    if tune:
        header = f"{'seq_len':<8} {'cfg(bM,bN,ns,th)':<20} {'Std (ms)':<11} {'GA (ms)':<10} {'Official':<10} {'sdpa':<10} {'Manual':<11} {'GA vs Std':<11} {'GA vs Off':<11}"
    else:
        header = f"{'seq_len':<10} {'Std (ms)':<11} {'GA (ms)':<10} {'Official':<10} {'sdpa':<10} {'Manual':<11} {'GA vs Std':<11} {'GA vs Off':<11}"
    print(header)
    print("-" * len(header))
    for r in results:
        ga_vs_std = r["mine_ms"] / r["mine_ga_ms"]
        ga_vs_off = r["official_ms"] / r["mine_ga_ms"]
        if tune:
            c = r["config"]
            cfg_str = f"({c['block_M']},{c['block_N']},{c['num_stages']},{c['threads']})"
            print(f"{r['seq_len']:<8} {cfg_str:<20} {r['mine_ms']:<11.4f} {r['mine_ga_ms']:<10.4f} {r['official_ms']:<10.4f} {r['sdpa_ms']:<10.4f} {r['manual_ms']:<11.4f} {ga_vs_std:<11.2f}x {ga_vs_off:<11.2f}x")
        else:
            print(f"{r['seq_len']:<10} {r['mine_ms']:<11.4f} {r['mine_ga_ms']:<10.4f} {r['official_ms']:<10.4f} {r['sdpa_ms']:<10.4f} {r['manual_ms']:<11.4f} {ga_vs_std:<11.2f}x {ga_vs_off:<11.2f}x")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GQA Flash Attention — Forward Pass")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv_heads", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--is_causal", action="store_true", default=False)
    parser.add_argument("--block_M", type=int, default=64)
    parser.add_argument("--block_N", type=int, default=64)
    parser.add_argument("--num_stages", type=int, default=2)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--sweep", action="store_true", default=False,
                        help="Run sweep benchmark across seq_lens 512,1024,2048,4096")
    parser.add_argument("--tune", action="store_true", default=False,
                        help="Enable autotuning over (block_M, block_N, num_stages, threads). "
                             "Overrides any --block_M / --block_N / --num_stages / --threads.")
    args = parser.parse_args()

    if args.sweep:
        benchmark_sweep(
            seq_lens=(512, 1024, 2048, 4096),
            batch=args.batch,
            heads=args.heads,
            kv_heads=args.kv_heads,
            dim=args.dim,
            is_causal=args.is_causal,
            block_M=args.block_M,
            block_N=args.block_N,
            num_stages=args.num_stages,
            threads=args.threads,
            tune=args.tune,
        )
    else:
        main(
            batch=args.batch,
            heads=args.heads,
            kv_heads=args.kv_heads,
            seq_len=args.seq_len,
            dim=args.dim,
            is_causal=args.is_causal,
            block_M=args.block_M,
            block_N=args.block_N,
            num_stages=args.num_stages,
            threads=args.threads,
            tune=args.tune,
        )
