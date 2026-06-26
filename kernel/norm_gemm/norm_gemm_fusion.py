import torch
import tilelang
import tilelang.language as T


@tilelang.jit
def norm_gemm_fusion(
    A,
    B,
    block_M: int = 256,
    block_N: int = 64,
    block_K: int = 64,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        local = T.alloc_fragment((block_M, block_K), accum_dtype)
        A_powsum = T.alloc_fragment((block_M,), accum_dtype)

        num_k_step = T.ceildiv(K, block_K)

        # Pass 1: accumulate x² across K, then reduce to per-row scale
        T.clear(local)
        for k in T.Serial(num_k_step):
            T.copy(A[bx * block_M, k * block_K], A_shared)
            for i, j in T.Parallel(block_M, block_K):
                local[i, j] += A_shared[i, j].astype(accum_dtype) * A_shared[i, j].astype(accum_dtype)

        T.reduce_sum(local, A_powsum, dim=1)
        for i in T.Parallel(block_M):
            A_powsum[i] = T.rsqrt(A_powsum[i] / K + 1e-5)

        # Pass 2: GEMM with post-scale (local reused as C_local)
        A_shared_2 = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        T.clear(local)
        for k in T.Pipelined(num_k_step, num_stages=2):
            T.copy(A[bx * block_M, k * block_K], A_shared_2)
            T.copy(B[k * block_K, by * block_N], B_shared)
            T.gemm(A_shared_2, B_shared, local)

        for i, j in T.Parallel(block_M, block_N):
            local[i, j] = local[i, j] * A_powsum[i]

        T.copy(local, C[bx * block_M, by * block_N])

    return C


# ============================================================
# Official TileLang RMSNorm (from examples/norm/rms_norm.py)
# Loads one row per block into fragment — single-pass, no shared memory tiling.
# ============================================================
@tilelang.jit(pass_configs={"tl.disable_tma_lower": True})
def rms_norm_only(A, blk_m: int = 1):
    M, N = T.const("M, N")
    dtype = T.float
    A: T.Tensor((M, N), dtype)
    B = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
        A_local = T.alloc_fragment((blk_m, N), dtype)
        A_pow_local = T.alloc_fragment((blk_m, N), dtype)
        A_powsum = T.alloc_fragment((blk_m,), dtype)

        T.copy(A[bx * blk_m : (bx + 1) * blk_m, :], A_local)
        for i, j in T.Parallel(blk_m, N):
            A_pow_local[i, j] = A_local[i, j] * A_local[i, j]
        T.reduce_sum(A_pow_local, A_powsum, dim=1)
        for i in T.Parallel(blk_m):
            A_powsum[i] = T.rsqrt(A_powsum[i] / N + 1e-12)
        for i, j in T.Parallel(blk_m, N):
            A_local[i, j] *= A_powsum[i]
        T.copy(A_local, B[bx * blk_m : (bx + 1) * blk_m, :])

    return B


# ============================================================
# Official TileLang GEMM (from examples/gemm/example_gemm.py)
# ============================================================
@tilelang.jit
def gemm_only(A, B, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


# ============================================================
# Reference implementation
# ============================================================
def ref_norm_gemm(x, w, eps=1e-5):
    """RMSNorm + GEMM in PyTorch (non-fused)."""
    rms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return rms @ w


# ============================================================
# Main: correctness + performance benchmark
# ============================================================
def main(M=8192, N=8192, K=8192, block_M=256, block_N=64, block_K=64):
    dtype = torch.float16
    device = "cuda"

    print(f"Config: M={M}, N={N}, K={K}, block_M={block_M}, block_N={block_N}, block_K={block_K}")
    print(f"dtype={dtype}")

    # Input data
    x = torch.randn(M, K, device=device, dtype=dtype)
    w = torch.randn(K, N, device=device, dtype=dtype)

    # ---- TileLang fused kernel ----
    kernel = norm_gemm_fusion.compile(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
    )
    c_tilelang = kernel(x, w)

    # ---- Reference ----
    c_ref = ref_norm_gemm(x.float(), w.float()).to(dtype)

    # ---- Correctness ----
    try:
        torch.testing.assert_close(c_tilelang, c_ref, rtol=1e-2, atol=1e-2)
        print("\n[PASS] TileLang fused kernel matches reference (rtol=1e-2)")
    except AssertionError as e:
        max_err = (c_tilelang.float() - c_ref.float()).abs().max().item()
        print(f"\n[FAIL] Mismatch! Max absolute error: {max_err:.6f}")
        print(e)

    # ---- Performance ----
    from tilelang.profiler import do_bench

    # Compile official TileLang standalone kernels
    # RMSNorm: official version uses blk_m=1 (one row per block, loads full row into fragment)
    rms_kernel = rms_norm_only.compile(M=M, N=K, blk_m=1)
    # GEMM: same block sizes as fused kernel for fair comparison
    gemm_kernel = gemm_only.compile(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
    )

    profiler = kernel.get_profiler()
    latency_fused = profiler.do_bench(warmup=25, rep=100)
    # Official RMSNorm uses fp32, pass float input
    latency_rms_tl = rms_kernel.get_profiler().do_bench(warmup=25, rep=100)
    latency_gemm_tl = gemm_kernel.get_profiler().do_bench(warmup=25, rep=100)

    # PyTorch baselines
    def ptx_rmsnorm(xx):
        return xx * torch.rsqrt(xx.float().pow(2).mean(-1, keepdim=True) + 1e-5).half()

    latency_rms_pt = do_bench(lambda: ptx_rmsnorm(x), warmup=25, rep=100)
    latency_mm_pt = do_bench(lambda: x @ w, warmup=25, rep=100)
    latency_unfused_pt = do_bench(lambda: ptx_rmsnorm(x) @ w, warmup=25, rep=100)

    print("\n--- Performance (ms) ---")
    print(f"TileLang Fused:     RMSNorm+GEMM  {latency_fused:.4f} ms")
    print(f"TileLang Non-Fused: RMSNorm only  {latency_rms_tl:.4f} ms")
    print(f"TileLang Non-Fused: GEMM only     {latency_gemm_tl:.4f} ms")
    print(f"TileLang Non-Fused: sum           {latency_rms_tl + latency_gemm_tl:.4f} ms")
    print(f"PyTorch:            RMSNorm only  {latency_rms_pt:.4f} ms")
    print(f"PyTorch:            GEMM only     {latency_mm_pt:.4f} ms")
    print(f"PyTorch:            RMSNorm+GEMM  {latency_unfused_pt:.4f} ms")
    print(f"PyTorch:            Non-Fus sum   {latency_rms_pt + latency_mm_pt:.4f} ms")
    print(f"Speedup vs TileLang non-fused:    {(latency_rms_tl + latency_gemm_tl) / latency_fused:.2f}x")
    print(f"Speedup vs PyTorch non-fused:     {latency_unfused_pt / latency_fused:.2f}x")

    # ---- CUDA source ----
    print("\nGenerated CUDA source saved to: /tmp/norm_gemm_fusion.cu")
    with open("/tmp/norm_gemm_fusion.cu", "w") as f:
        f.write(kernel.get_kernel_source())

    return kernel


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RMSNorm + GEMM Fusion")
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=8192)
    parser.add_argument("--K", type=int, default=8192)
    parser.add_argument("--block_M", type=int, default=256)
    parser.add_argument("--block_N", type=int, default=64)
    parser.add_argument("--block_K", type=int, default=64)
    args = parser.parse_args()

    main(
        M=args.M,
        N=args.N,
        K=args.K,
        block_M=args.block_M,
        block_N=args.block_N,
        block_K=args.block_K,
    )
