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

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench

# Import TileLang's official GQA Flash Attention for comparison
_examples_dir = os.path.join(os.path.dirname(tilelang.__file__), "..", "examples", "flash_attention")
if _examples_dir not in sys.path:
    sys.path.insert(0, _examples_dir)
from example_gqa_fwd_bshd import flashattn as official_gqa_flashattn


# =============================================================================
# GQA Flash Attention Kernel
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
    """
    GQA Flash Attention forward pass.

    Parameters
    ----------
    batch : int
        Number of independent sequences.
    heads : int
        Number of query heads (must be a multiple of kv_heads).
    kv_heads : int
        Number of key/value heads (heads // kv_heads = num_groups).
    seq_len : int
        Sequence length (same for Q and KV in self-attention).
    dim : int
        Head dimension (typically 64 or 128).
    is_causal : bool
        If True, apply causal mask (token i can only see tokens ≤ i).
    block_M : int
        Q row tile size (number of query rows per block).
    block_N : int
        KV column tile size (number of key/value columns per block).
    num_stages : int
        Software pipeline depth (1, 2, or 3).
    threads : int
        Number of threads per block.
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

                # ---- 4b. Apply mask before gemm ----
                # TODO: Fill S_local with the appropriate mask values.
                #   - If causal: position (i, j) maps to global (q_idx, k_idx).
                #     Set to 0 if q_idx >= k_idx, else -inf.
                #   - If non-causal: mask out padding positions where
                #     k * block_N + j >= seq_len.
                #   - Use T.Parallel(block_M, block_N) to iterate.
                #   - Use T.if_then_else(cond, true_val, false_val).
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
                # S_local is ACCUMULATED into: S_local += Q_shared @ K_shared^T
                # This means your mask values (0 or -inf) set the base,
                # and the dot products are added on top.
                T.gemm(Q_shared, K_shared, S_local, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # ============================================================
                # TODO: ONLINE SOFTMAX — implement the 6-step rescaling logic
                # ============================================================
                #
                # You need to:
                #
                #   (1) Save old max, reset S_max:
                #         T.copy(S_max, S_max_prev)
                #         T.fill(S_max, -T.infinity(accum_dtype))
                #
                #   (2) Compute row-wise max of the NEW S_local block:
                #         T.reduce_max(S_local, S_max, dim=1, clear=False)
                #
                #   (3) Update running max:
                #         S_max[i] = T.max(S_max[i], S_max_prev[i])
                #         (use T.Parallel(block_M))
                #
                #   (4) Compute rescale factor:
                #         rescale[i] = T.exp2(
                #             S_max_prev[i] * scale - S_max[i] * scale
                #         )
                #
                #   (5) Rescale old O_local BEFORE adding new contributions:
                #         O_local[i, j] *= rescale[i]
                #         (use T.Parallel(block_M, dim))
                #
                #   (6) Compute softmax numerators P = exp(S - max):
                #         S_local[i, j] = T.exp2(
                #             S_local[i, j] * scale - S_max[i] * scale
                #         )
                #         (use T.Parallel(block_M, block_N))
                #         This REPLACES S_local in-place — S is overwritten by P.
                #
                #   (7) Update total_sum:
                #         T.reduce_sum(S_local, S_exp_sum, dim=1)
                #         total_sum[i] = total_sum[i] * rescale[i] + S_exp_sum[i]
                #         (use T.Parallel(block_M))
                #
                #   (8) Cast P from fp32 to fp16 for the next gemm:
                #         T.copy(S_local, S_local_fp16)
                #
                # Key insight: scale = 1/sqrt(dim) * log2(e).
                #   exp2(x * scale) = exp(x / sqrt(dim)).
                # The rescale factor exp2((m_old - m_new) * scale) equals
                # exp((m_old - m_new) / sqrt(dim)) — corrects old softmax
                # numerators to use the new, larger max.
                #
                # ============================================================

                T.copy(S_max, S_max_prev)
                T.fill(S_max, -T.infinity(accum_dtype))

                T.reduce_max(S_local, S_max, dim=1, clear=False)

                for i in T.Parallel(block_M):
                    S_max[i] = T.max(S_max[i], S_max_prev[i])
                    rescale[i] = T.exp2(S_max_prev[i] * scale - S_max[i] * scale)

                for i, j in T.Parallel(block_M, block_N):
                    S_local[i, j] = T.exp2(S_local[i, j] * scale - S_max[i] * scale)

                T.reduce_sum(S_local, S_exp_sum, dim=1)
                for i in T.Parallel(block_M):
                    total_sum[i] = total_sum[i] * rescale[i] + S_exp_sum[i]

                for i, j in T.Parallel(block_M, dim):
                    O_local[i, j] *= rescale[i]

                T.copy(S_local, S_local_fp16)
                # ---- 4d. Load V and accumulate P @ V → O_local ----
                T.copy(V[bz, k * block_N : (k + 1) * block_N, kv_head, :], V_shared)
                T.gemm(S_local_fp16, V_shared, O_local, policy=T.GemmWarpPolicy.FullRow)

            # ---- Step 5: Final normalization ----
            # Divide each row of the accumulated output by its softmax denominator.
            # Normalize the output by the total sum.
            # TODO: implement the normalization
            for i, j in T.Parallel(block_M, dim):
                O_local[i, j] /= total_sum[i]

            # ---- Step 6: Write output back to HBM ----
            # Route through O_shared for coalesced memory writes.
            T.copy(O_local, O_shared)
            T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return main


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
):
    dtype = torch.float16
    device = "cuda"

    print("=" * 60)
    print("GQA Flash Attention — Forward Pass")
    print("=" * 60)
    print(f"Layout: BSHD  |  batch={batch}, heads={heads}, kv_heads={kv_heads}")
    print(f"seq_len={seq_len}, dim={dim}, causal={is_causal}")
    print(f"block_M={block_M}, block_N={block_N}, num_stages={num_stages}, threads={threads}")

    # ---- Input data ----
    Q = torch.randn(batch, seq_len, heads, dim, device=device, dtype=dtype)
    K = torch.randn(batch, seq_len, kv_heads, dim, device=device, dtype=dtype)
    V = torch.randn(batch, seq_len, kv_heads, dim, device=device, dtype=dtype)

    # ---- Compile TileLang kernel ----
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

    # ---- Correctness ----
    result = kernel(Q, K, V)
    reference = ref_gqa(Q, K, V, is_causal=is_causal)

    try:
        torch.testing.assert_close(result, reference, rtol=1e-2, atol=1e-2)
        print("\n[PASS] TileLang kernel matches PyTorch reference (rtol=1e-2)")
    except AssertionError as e:
        max_err = (result.float() - reference.float()).abs().max().item()
        rel_err = ((result.float() - reference.float()).abs() / (reference.float().abs() + 1e-8)).max().item()
        print(f"\n[FAIL] Mismatch!  Max absolute error: {max_err:.6f}, Max relative error: {rel_err:.6f}")
        print(e)

    # ---- Generated CUDA source ----
    src_path = f"/tmp/gqa_flash_attn_s{seq_len}_d{dim}.cu"
    with open(src_path, "w") as f:
        f.write(kernel.get_kernel_source())
    print(f"\nCUDA source saved to: {src_path}")

    # ---- Performance ----
    num_groups = heads // kv_heads

    # Compile TileLang's official GQA Flash Attention with matching params
    # The official API uses `groups` (Q heads per KV head) instead of `kv_heads`
    kernel_official = official_gqa_flashattn(
        batch,
        heads,
        seq_len,
        dim,
        is_causal,
        groups=num_groups,
        block_M=block_M,
        block_N=block_N,
        num_stages=num_stages,
        threads=threads,
    )

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

    lat_mine = do_bench(lambda: kernel(Q, K, V), warmup=25, rep=100)
    lat_official = do_bench(lambda: kernel_official(Q, K, V), warmup=25, rep=100)
    lat_sdpa = do_bench(lambda: ptx_sdpa(Q, K, V, is_causal), warmup=25, rep=100)
    lat_manual = do_bench(lambda: ptx_manual(Q, K, V, is_causal), warmup=25, rep=100)

    flops = 2.0 * batch * heads * seq_len * seq_len * dim * 2  # 2 matmuls (QK^T + PV)
    if is_causal:
        flops *= 0.5

    print("\n--- Performance ---")
    print(f"{'Backend':<35} {'Latency (ms)':<15} {'TFlops':<10}")
    print("-" * 60)
    print(f"{'My GQA Flash Attn':<35} {lat_mine:<15.4f} {flops / lat_mine * 1e-9:<10.2f}")
    print(f"{'TileLang Official GQA':<35} {lat_official:<15.4f} {flops / lat_official * 1e-9:<10.2f}")
    print(f"{'PyTorch sdpa (cuDNN FA)':<35} {lat_sdpa:<15.4f} {flops / lat_sdpa * 1e-9:<10.2f}")
    print(f"{'PyTorch manual (no fusion)':<35} {lat_manual:<15.4f} {flops / lat_manual * 1e-9:<10.2f}")
    print(f"\nMy impl vs Official:     {lat_official / lat_mine:.2f}x")
    print(f"My impl vs PyTorch sdpa:  {lat_sdpa / lat_mine:.2f}x")
    print(f"My impl vs PyTorch manual:{lat_manual / lat_mine:.2f}x")

    return kernel


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
    args = parser.parse_args()

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
    )
