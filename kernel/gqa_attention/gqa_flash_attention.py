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
# Resolve relative to THIS file so it works on any machine
_examples_dir = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "flash_attention")
_examples_dir = os.path.abspath(_examples_dir)
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

    lat_mine = do_bench(lambda: kernel(Q, K, V), warmup=500, rep=100)
    lat_official = do_bench(lambda: kernel_official(Q, K, V), warmup=500, rep=100)
    lat_sdpa = do_bench(lambda: ptx_sdpa(Q, K, V, is_causal), warmup=500, rep=100)
    lat_manual = do_bench(lambda: ptx_manual(Q, K, V, is_causal), warmup=500, rep=100)

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
):
    """Run correctness + performance across multiple sequence lengths."""
    dtype = torch.float16
    device = "cuda"
    num_groups = heads // kv_heads

    results = []
    print("=" * 80)
    print(f"GQA Flash Attention — Sweep: seq_len ∈ {list(seq_lens)}")
    print(f"batch={batch}, heads={heads}, kv_heads={kv_heads}, dim={dim}, causal={is_causal}")
    print("=" * 80)

    for seq_len in seq_lens:
        print(f"\n--- seq_len = {seq_len} ---")

        Q = torch.randn(batch, seq_len, heads, dim, device=device, dtype=dtype)
        K = torch.randn(batch, seq_len, kv_heads, dim, device=device, dtype=dtype)
        V = torch.randn(batch, seq_len, kv_heads, dim, device=device, dtype=dtype)

        # My kernel
        kernel = gqa_flash_attn(
            batch=batch, heads=heads, kv_heads=kv_heads,
            seq_len=seq_len, dim=dim, is_causal=is_causal,
            block_M=block_M, block_N=block_N,
            num_stages=num_stages, threads=threads,
        )

        # Official kernel
        kernel_official = official_gqa_flashattn(
            batch, heads, seq_len, dim, is_causal,
            groups=num_groups,
            block_M=block_M, block_N=block_N,
            num_stages=num_stages, threads=threads,
        )

        # Correctness
        result = kernel(Q, K, V)
        reference = ref_gqa(Q, K, V, is_causal=is_causal)
        try:
            torch.testing.assert_close(result, reference, rtol=1e-2, atol=1e-2)
            print("  [PASS] correctness")
        except AssertionError as e:
            max_err = (result.float() - reference.float()).abs().max().item()
            print(f"  [FAIL] max error = {max_err:.6f}")
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
        lat_official = do_bench(lambda: kernel_official(Q, K, V), warmup=500, rep=100)
        lat_sdpa = do_bench(lambda: ptx_sdpa(Q, K, V), warmup=500, rep=100)
        lat_manual = do_bench(lambda: ref_gqa(Q, K, V, is_causal=is_causal), warmup=500, rep=100)

        flops = 4.0 * batch * heads * seq_len * seq_len * dim  # QK^T + PV
        if is_causal:
            flops *= 0.5

        results.append({
            "seq_len": seq_len,
            "mine_ms": lat_mine,
            "mine_tflops": flops / lat_mine * 1e-9,
            "official_ms": lat_official,
            "sdpa_ms": lat_sdpa,
            "manual_ms": lat_manual,
        })

        print(f"  Mine: {lat_mine:.4f} ms  |  Official: {lat_official:.4f} ms  |  sdpa: {lat_sdpa:.4f} ms  |  Manual: {lat_manual:.4f} ms")

    # Print summary table
    print("\n" + "=" * 80)
    print("Summary Table")
    print("=" * 80)
    header = f"{'seq_len':<10} {'Mine (ms)':<12} {'Official (ms)':<14} {'sdpa (ms)':<12} {'Manual (ms)':<13} {'vs Official':<12} {'vs Manual':<12}"
    print(header)
    print("-" * len(header))
    for r in results:
        vs_official = r["official_ms"] / r["mine_ms"]
        vs_manual = r["manual_ms"] / r["mine_ms"]
        print(f"{r['seq_len']:<10} {r['mine_ms']:<12.4f} {r['official_ms']:<14.4f} {r['sdpa_ms']:<12.4f} {r['manual_ms']:<13.4f} {vs_official:<12.2f}x {vs_manual:<12.2f}x")

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
        )
