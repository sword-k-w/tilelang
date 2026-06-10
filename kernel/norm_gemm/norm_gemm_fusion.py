import torch
import tilelang
import tilelang.language as T


@tilelang.jit(pass_configs={"tl.disable_tma_lower": True})
def norm_gemm_fusion(
    A,
    B,
    block_M: int = 32,
    block_N: int = 32,
    block_K: int = 32,
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

        # ==========================================================
        # Pass 1: compute scale[i] = rsqrt(mean(A[i,:]^2) + eps)
        # ==========================================================
        num_k_step = T.ceildiv(K, block_K)
        T.clear(local)

        for k in T.Serial(num_k_step):
            T.copy(A[bx * block_M, k * block_K], A_shared)
            for i, j in T.Parallel(block_M, block_K):
                local[i, j] += A_shared[i, j].astype(accum_dtype) * A_shared[i, j].astype(accum_dtype)

        T.reduce_sum(local, A_powsum, dim=1)
        for i in T.Parallel(block_M):
            A_powsum[i] = T.rsqrt(A_powsum[i] / K + 1e-5)

        # ==========================================================
        # Pass 2: normalized GEMM
        # ==========================================================
        A_shared_2 = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        # C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        T.clear(local)

        for k in T.Pipelined(num_k_step, num_stages=3):
            T.copy(A[bx * block_M, k * block_K], A_shared_2)
            T.copy(B[k * block_K, by * block_N], B_shared)
            T.gemm(A_shared_2, B_shared, local)

        for i, j in T.Parallel(block_M, block_N):
            local[i, j] = local[i, j] * A_powsum[i]

        T.copy(local, C[bx * block_M, by * block_N])

    return C


# ============================================================
# Reference implementation (unfused)
# ============================================================
def ref_norm_gemm(x, w, eps=1e-5):
    """RMSNorm + GEMM in PyTorch (non-fused)."""
    rms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return rms @ w


# ============================================================
# Main: correctness + performance benchmark
# ============================================================
def main(M=4096, N=4096, K=4096, block_M=32, block_N=32, block_K=32):
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

    profiler = kernel.get_profiler()
    latency_fused = profiler.do_bench(warmup=25, rep=100)

    # PyTorch baselines
    def ptx_rmsnorm(xx):
        return xx * torch.rsqrt(xx.float().pow(2).mean(-1, keepdim=True) + 1e-5).half()

    latency_rms = do_bench(lambda: ptx_rmsnorm(x), warmup=25, rep=100)
    latency_mm = do_bench(lambda: x @ w, warmup=25, rep=100)
    latency_unfused = do_bench(lambda: ptx_rmsnorm(x) @ w, warmup=25, rep=100)

    print("\n--- Performance (ms) ---")
    print(f"TileLang RMSNorm+GEMM (fused):  {latency_fused:.4f} ms")
    print(f"PyTorch  RMSNorm only:           {latency_rms:.4f} ms")
    print(f"PyTorch  GEMM only:              {latency_mm:.4f} ms")
    print(f"PyTorch  RMSNorm+GEMM (non-fus): {latency_unfused:.4f} ms")
    print(f"PyTorch  Non-Fused sum:          {latency_rms + latency_mm:.4f} ms")
    print(f"Speedup vs PyTorch non-fused:    {latency_unfused / latency_fused:.2f}x")
    print(f"Speedup vs sum:                  {(latency_rms + latency_mm) / latency_fused:.2f}x")

    # ---- CUDA source ----
    print("\nGenerated CUDA source saved to: /tmp/norm_gemm_fusion.cu")
    with open("/tmp/norm_gemm_fusion.cu", "w") as f:
        f.write(kernel.get_kernel_source())

    return kernel


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RMSNorm + GEMM Fusion")
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--block_M", type=int, default=32)
    parser.add_argument("--block_N", type=int, default=32)
    parser.add_argument("--block_K", type=int, default=32)
    args = parser.parse_args()

    main(
        M=args.M,
        N=args.N,
        K=args.K,
        block_M=args.block_M,
        block_N=args.block_N,
        block_K=args.block_K,
    )
