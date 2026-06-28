"""
Triton GQA Flash Attention — Forward Pass (BSHD layout)
========================================================

Drop-in benchmark companion to ``gqa_flash_attention.py``.

Provides two Triton kernels:
  - ``triton_gqa_fwd``      — standard GQA (one block per Q head)
  - ``triton_gqa_fwd_ga``   — group-aware (one block per KV head, K/V loaded once)

Usage (standalone):
    python kernel/gqa_attention/triton_gqa_bench.py
    python kernel/gqa_attention/triton_gqa_bench.py --seq_len 2048 --is_causal
"""

import argparse

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ==============================================================================
# Standard Triton GQA Flash Attention Kernel
# ==============================================================================
@triton.jit
def _triton_gqa_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    Out_ptr,
    stride_qb,
    stride_qs,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_os,
    stride_oh,
    stride_od,
    seq_len: tl.constexpr,      # int, passed as value so it can be used in range()
    dim: tl.constexpr,
    heads: tl.constexpr,
    kv_heads: tl.constexpr,
    is_causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Standard GQA: one program per (Q row block, Q head, batch element).

    Grid: (cdiv(seq_len, BLOCK_M), batch * heads)
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_idx = pid_bh // heads
    head_idx = pid_bh % heads

    num_groups = heads // kv_heads
    kv_head_idx = head_idx // num_groups

    # ---- Offsets ----
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_d = tl.arange(0, dim)                         # [dim]

    # ---- Pointer blocks ----
    # Q[batch_idx, offs_m, head_idx, offs_d]
    Q_block = (
        Q_ptr
        + batch_idx * stride_qb
        + offs_m[:, None] * stride_qs
        + head_idx * stride_qh
        + offs_d[None, :] * stride_qd
    )  # [BLOCK_M, dim]
    q = tl.load(Q_block, mask=(offs_m[:, None] < seq_len), other=0.0)

    # ---- Running state (fp32) ----
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, dim], dtype=tl.float32)

    scale = 1.0 / (dim**0.5)

    # ---- Main KV loop ----
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)  # [BLOCK_N]

        # K[batch_idx, offs_n, kv_head_idx, offs_d]
        K_block = (
            K_ptr
            + batch_idx * stride_kb
            + offs_n[:, None] * stride_ks
            + kv_head_idx * stride_kh
            + offs_d[None, :] * stride_kd
        )  # [BLOCK_N, dim]
        k = tl.load(K_block, mask=(offs_n[:, None] < seq_len), other=0.0)

        # S = Q @ K^T  →  [BLOCK_M, BLOCK_N]
        s = tl.dot(q, tl.trans(k)) * scale

        # ---- Mask ----
        if is_causal:
            s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float("-inf"))
        # Mask invalid KV positions
        s = tl.where(offs_n[None, :] < seq_len, s, float("-inf"))

        # ---- Online softmax (FlashAttention-1 Algorithm 1) ----
        m_ij = tl.max(s, axis=1)                 # [BLOCK_M]
        m_new = tl.maximum(m_i, m_ij)            # [BLOCK_M]

        # P = exp(S - m_new)  *BUT* we need to rescale old accumulator
        alpha = tl.exp(m_i - m_new)               # [BLOCK_M]
        p = tl.exp(s - m_new[:, None])             # [BLOCK_M, BLOCK_N]

        # Rescale old accumulator
        acc = acc * alpha[:, None]                 # [BLOCK_M, dim]
        l_i = l_i * alpha + tl.sum(p, axis=1)      # [BLOCK_M]

        # ---- Load V and accumulate P @ V ----
        V_block = (
            V_ptr
            + batch_idx * stride_vb
            + offs_n[:, None] * stride_vs
            + kv_head_idx * stride_vh
            + offs_d[None, :] * stride_vd
        )  # [BLOCK_N, dim]
        v = tl.load(V_block, mask=(offs_n[:, None] < seq_len), other=0.0)

        acc += tl.dot(p.to(tl.float16), v)

        m_i = m_new

    # ---- Final normalization ----
    acc = acc / l_i[:, None]

    # ---- Write output ----
    Out_block = (
        Out_ptr
        + batch_idx * stride_ob
        + offs_m[:, None] * stride_os
        + head_idx * stride_oh
        + offs_d[None, :] * stride_od
    )  # [BLOCK_M, dim]
    tl.store(Out_block, acc.to(tl.float16), mask=(offs_m[:, None] < seq_len))


# ==============================================================================
# Group-Aware Triton GQA Kernel
# ==============================================================================
@triton.jit
def _triton_gqa_fwd_ga_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    Out_ptr,
    stride_qb,
    stride_qs,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_os,
    stride_oh,
    stride_od,
    seq_len: tl.constexpr,
    dim: tl.constexpr,
    heads: tl.constexpr,
    kv_heads: tl.constexpr,
    num_groups: tl.constexpr,
    is_causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Group-aware GQA: one program per (Q row block, KV head, batch element).

    Grid: (cdiv(seq_len, BLOCK_M), batch * kv_heads)

    Within each block, K/V are loaded **once** per KV iteration and reused
    across all ``num_groups`` Q heads sharing this KV head.  Q data for all
    groups is loaded as a single [M_TILE, dim] tile using per-row head-index
    pointer arithmetic, and softmax state is maintained as [M_TILE, ...]
    tensors where rows of different groups are independent.
    """
    pid_m = tl.program_id(0)
    pid_bkv = tl.program_id(1)

    batch_idx = pid_bkv // kv_heads
    kv_head_idx = pid_bkv % kv_heads

    # ---- Offsets ----
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_d = tl.arange(0, dim)                         # [dim]

    scale = 1.0 / (dim**0.5)

    # ---- Per-group running state, packed as [M_TILE, ...] ----
    M_TILE: tl.constexpr = num_groups * BLOCK_M

    m_all = tl.full([M_TILE], float("-inf"), dtype=tl.float32)
    l_all = tl.zeros([M_TILE], dtype=tl.float32)
    acc_all = tl.zeros([M_TILE, dim], dtype=tl.float32)

    # ---- Row metadata ----
    offs_m_tile = tl.arange(0, M_TILE)          # [M_TILE]
    group_of_row = offs_m_tile // BLOCK_M       # [M_TILE], which group this row belongs to
    s_of_row = offs_m_tile % BLOCK_M             # [M_TILE], seq position within the Q block

    # ---- Load Q for all groups into q_all: [M_TILE, dim] ----
    # Each row i loads from Q head = kv_head * num_groups + group_of_row[i]
    head_of_row = kv_head_idx * num_groups + group_of_row  # [M_TILE]
    q_ptrs = (
        Q_ptr
        + batch_idx * stride_qb
        + (pid_m * BLOCK_M + s_of_row)[:, None] * stride_qs
        + head_of_row[:, None] * stride_qh
        + offs_d[None, :] * stride_qd
    )  # [M_TILE, dim]
    q_all = tl.load(q_ptrs, mask=((pid_m * BLOCK_M + s_of_row)[:, None] < seq_len), other=0.0)

    # ---- Main KV loop ----
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)  # [BLOCK_N]

        # Load K once for all groups
        K_block = (
            K_ptr
            + batch_idx * stride_kb
            + offs_n[:, None] * stride_ks
            + kv_head_idx * stride_kh
            + offs_d[None, :] * stride_kd
        )  # [BLOCK_N, dim]
        k = tl.load(K_block, mask=(offs_n[:, None] < seq_len), other=0.0)

        # S_all = Q_all @ K^T → [M_TILE, BLOCK_N]
        s_all = tl.dot(q_all, tl.trans(k)) * scale

        # ---- Mask: causal and out-of-bounds ----
        # Absolute Q position for each row
        q_abs = pid_m * BLOCK_M + s_of_row  # [M_TILE]
        if is_causal:
            s_all = tl.where(q_abs[:, None] >= offs_n[None, :], s_all, float("-inf"))
        s_all = tl.where(offs_n[None, :] < seq_len, s_all, float("-inf"))

        # ---- Online softmax (per-row, rows of different groups are independent) ----
        m_ij = tl.max(s_all, axis=1)              # [M_TILE]
        m_new = tl.maximum(m_all, m_ij)           # [M_TILE]

        alpha = tl.exp(m_all - m_new)             # [M_TILE]
        p_all = tl.exp(s_all - m_new[:, None])    # [M_TILE, BLOCK_N]

        # Rescale old accumulator
        acc_all = acc_all * alpha[:, None]
        l_all = l_all * alpha + tl.sum(p_all, axis=1)

        # Load V once, accumulate P @ V
        V_block = (
            V_ptr
            + batch_idx * stride_vb
            + offs_n[:, None] * stride_vs
            + kv_head_idx * stride_vh
            + offs_d[None, :] * stride_vd
        )  # [BLOCK_N, dim]
        v = tl.load(V_block, mask=(offs_n[:, None] < seq_len), other=0.0)

        acc_all += tl.dot(p_all.to(tl.float16), v)
        m_all = m_new

    # ---- Final normalization ----
    acc_all = acc_all / l_all[:, None]  # [M_TILE, dim]

    # ---- Write output: each row goes to its Q head ----
    out_ptrs = (
        Out_ptr
        + batch_idx * stride_ob
        + (pid_m * BLOCK_M + s_of_row)[:, None] * stride_os
        + head_of_row[:, None] * stride_oh
        + offs_d[None, :] * stride_od
    )  # [M_TILE, dim]
    tl.store(
        out_ptrs,
        acc_all.to(tl.float16),
        mask=((pid_m * BLOCK_M + s_of_row)[:, None] < seq_len),
    )


# ==============================================================================
# Python wrappers (callable like the TileLang kernels)
# ==============================================================================


def triton_gqa_fwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool = False,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
) -> torch.Tensor:
    """Standard Triton GQA Flash Attention forward.

    Args:
        Q: [batch, seq_len, heads, dim] float16
        K: [batch, seq_len, kv_heads, dim] float16
        V: [batch, seq_len, kv_heads, dim] float16
        is_causal: apply causal mask
        BLOCK_M, BLOCK_N: tile sizes

    Returns:
        Out: [batch, seq_len, heads, dim] float16
    """
    batch, seq_len, heads, dim = Q.shape
    kv_heads = K.shape[2]
    assert heads % kv_heads == 0, f"heads ({heads}) must be divisible by kv_heads ({kv_heads})"
    assert K.shape[2] == kv_heads and V.shape[2] == kv_heads
    assert Q.shape[3] == dim and K.shape[3] == dim and V.shape[3] == dim

    Out = torch.empty_like(Q)

    # Strides for BSHD contiguous tensors
    stride_qb, stride_qs, stride_qh, stride_qd = Q.stride()
    stride_kb, stride_ks, stride_kh, stride_kd = K.stride()
    stride_vb, stride_vs, stride_vh, stride_vd = V.stride()
    stride_ob, stride_os, stride_oh, stride_od = Out.stride()

    grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads)

    _triton_gqa_fwd_kernel[grid](
        Q,
        K,
        V,
        Out,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vs,
        stride_vh,
        stride_vd,
        stride_ob,
        stride_os,
        stride_oh,
        stride_od,
        seq_len=seq_len,
        dim=dim,
        heads=heads,
        kv_heads=kv_heads,
        is_causal=is_causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )
    return Out


def triton_gqa_fwd_ga(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool = False,
    BLOCK_M: int = 32,
    BLOCK_N: int = 64,
) -> torch.Tensor:
    """Group-aware Triton GQA Flash Attention forward.

    K/V are loaded once per KV-head block and reused across all Q heads
    in the same group, reducing HBM traffic by ``num_groups``× for K and V.

    Args:
        Q: [batch, seq_len, heads, dim] float16
        K: [batch, seq_len, kv_heads, dim] float16
        V: [batch, seq_len, kv_heads, dim] float16
        is_causal: apply causal mask
        BLOCK_M: per-Q-head tile size (M_tile = num_groups * BLOCK_M)
        BLOCK_N: KV tile size

    Returns:
        Out: [batch, seq_len, heads, dim] float16
    """
    batch, seq_len, heads, dim = Q.shape
    kv_heads = K.shape[2]
    num_groups = heads // kv_heads
    assert heads % kv_heads == 0

    Out = torch.empty_like(Q)

    stride_qb, stride_qs, stride_qh, stride_qd = Q.stride()
    stride_kb, stride_ks, stride_kh, stride_kd = K.stride()
    stride_vb, stride_vs, stride_vh, stride_vd = V.stride()
    stride_ob, stride_os, stride_oh, stride_od = Out.stride()

    grid = (triton.cdiv(seq_len, BLOCK_M), batch * kv_heads)

    _triton_gqa_fwd_ga_kernel[grid](
        Q,
        K,
        V,
        Out,
        stride_qb,
        stride_qs,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vs,
        stride_vh,
        stride_vd,
        stride_ob,
        stride_os,
        stride_oh,
        stride_od,
        seq_len=seq_len,
        dim=dim,
        heads=heads,
        kv_heads=kv_heads,
        num_groups=num_groups,
        is_causal=is_causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )
    return Out


# ==============================================================================
# PyTorch reference
# ==============================================================================


def ref_gqa(Q, K, V, is_causal: bool = False):
    """Reference GQA via PyTorch (materialises full [seq, seq] attention)."""
    batch, seq_len, heads, dim = Q.shape
    kv_heads = K.shape[2]
    num_groups = heads // kv_heads

    K_exp = K.repeat_interleave(num_groups, dim=2)
    V_exp = V.repeat_interleave(num_groups, dim=2)

    Q_bhsd = Q.permute(0, 2, 1, 3)
    K_bhsd = K_exp.permute(0, 2, 1, 3)
    V_bhsd = V_exp.permute(0, 2, 1, 3)

    scores = torch.einsum("bhqd,bhkd->bhqk", Q_bhsd, K_bhsd) / (dim**0.5)
    if is_causal:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out = torch.einsum("bhqk,bhkd->bhqd", attn, V_bhsd)
    return out.permute(0, 2, 1, 3)


# ==============================================================================
# Benchmarking helpers
# ==============================================================================


def _do_bench(fn, warmup=100, rep=200):
    """Simple GPU-timed benchmark returning median latency in ms."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(rep):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return float(torch.tensor(times).median())


# ==============================================================================
# Standalone main
# ==============================================================================


def main(
    batch: int = 1,
    heads: int = 32,
    kv_heads: int = 8,
    seq_len: int = 1024,
    dim: int = 64,
    is_causal: bool = False,
):
    dtype = torch.float16
    device = "cuda"

    print("=" * 60)
    print("Triton GQA Flash Attention — Forward Pass")
    print("=" * 60)
    print(f"Layout: BSHD  |  batch={batch}, heads={heads}, kv_heads={kv_heads}")
    print(f"seq_len={seq_len}, dim={dim}, causal={is_causal}")

    Q = torch.randn(batch, seq_len, heads, dim, device=device, dtype=dtype)
    K = torch.randn(batch, seq_len, kv_heads, dim, device=device, dtype=dtype)
    V = torch.randn(batch, seq_len, kv_heads, dim, device=device, dtype=dtype)

    # ---- Standard Triton kernel ----
    out_std = triton_gqa_fwd(Q, K, V, is_causal=is_causal)
    ref = ref_gqa(Q, K, V, is_causal=is_causal)
    try:
        torch.testing.assert_close(out_std, ref, rtol=1e-2, atol=1e-2)
        print("\n[PASS] Triton standard kernel matches PyTorch reference (rtol=1e-2)")
    except AssertionError as e:
        max_err = (out_std.float() - ref.float()).abs().max().item()
        rel_err = ((out_std.float() - ref.float()).abs() / (ref.float().abs() + 1e-8)).max().item()
        print(f"\n[FAIL] Triton standard mismatch!  Max abs err: {max_err:.6f}, Max rel err: {rel_err:.6f}")
        print(e)

    # ---- Group-aware Triton kernel ----
    out_ga = triton_gqa_fwd_ga(Q, K, V, is_causal=is_causal)
    try:
        torch.testing.assert_close(out_ga, ref, rtol=1e-2, atol=1e-2)
        print("[PASS] Triton group-aware kernel matches PyTorch reference (rtol=1e-2)")
    except AssertionError as e:
        max_err = (out_ga.float() - ref.float()).abs().max().item()
        rel_err = ((out_ga.float() - ref.float()).abs() / (ref.float().abs() + 1e-8)).max().item()
        print(f"[FAIL] Triton group-aware mismatch!  Max abs err: {max_err:.6f}, Max rel err: {rel_err:.6f}")
        print(e)

    # ---- Benchmark ----
    lat_std = _do_bench(lambda: triton_gqa_fwd(Q, K, V, is_causal=is_causal))
    lat_ga = _do_bench(lambda: triton_gqa_fwd_ga(Q, K, V, is_causal=is_causal))
    lat_ref = _do_bench(lambda: ref_gqa(Q, K, V, is_causal=is_causal))

    # PyTorch sdpa
    def ptx_sdpa(q, k, v, causal):
        q_b = q.permute(0, 2, 1, 3)
        k_b = k.permute(0, 2, 1, 3)
        v_b = v.permute(0, 2, 1, 3)
        ng = q.shape[2] // k.shape[2]
        k_be = k_b.repeat_interleave(ng, dim=1)
        v_be = v_b.repeat_interleave(ng, dim=1)
        o = F.scaled_dot_product_attention(q_b, k_be, v_be, is_causal=causal)
        return o.permute(0, 2, 1, 3)

    lat_sdpa = _do_bench(lambda: ptx_sdpa(Q, K, V, is_causal))

    flops = 4.0 * batch * heads * seq_len * seq_len * dim
    if is_causal:
        flops *= 0.5

    print("\n--- Performance ---")
    print(f"{'Backend':<35} {'Latency (ms)':<15} {'TFlops':<10}")
    print("-" * 60)
    print(f"{'Triton GQA (standard)':<35} {lat_std:<15.4f} {flops / lat_std * 1e-9:<10.2f}")
    print(f"{'Triton GQA (group-aware)':<35} {lat_ga:<15.4f} {flops / lat_ga * 1e-9:<10.2f}")
    print(f"{'PyTorch sdpa (cuDNN FA)':<35} {lat_sdpa:<15.4f} {flops / lat_sdpa * 1e-9:<10.2f}")
    print(f"{'PyTorch manual (no fusion)':<35} {lat_ref:<15.4f} {flops / lat_ref * 1e-9:<10.2f}")

    return out_std


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton GQA Flash Attention")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv_heads", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--is_causal", action="store_true", default=False)
    args = parser.parse_args()

    main(
        batch=args.batch,
        heads=args.heads,
        kv_heads=args.kv_heads,
        seq_len=args.seq_len,
        dim=args.dim,
        is_causal=args.is_causal,
    )
