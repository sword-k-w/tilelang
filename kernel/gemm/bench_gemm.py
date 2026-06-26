"""Benchmark TileLang GEMM vs PyTorch (cuBLAS).

Mimics the official example (examples/gemm/example_gemm.py) and allows
flexible parameter tuning like kernel/norm_gemm/norm_gemm_fusion.py.
"""

import argparse

import torch

import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


# ============================================================
# TileLang GEMM  (matches examples/gemm/example_gemm.py)
# ============================================================
@tilelang.jit
def matmul(
    A,
    B,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
):
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
# Main: correctness + performance
# ============================================================
def main(M=4096, N=4096, K=4096, block_M=128, block_N=128, block_K=32):
    dtype = torch.float16
    device = "cuda"

    print(f"Config: M={M}, N={N}, K={K}, block_M={block_M}, block_N={block_N}, block_K={block_K}")
    print(f"dtype={dtype}")

    # Input data
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)

    # ---- Compile TileLang kernel ----
    kernel = matmul.compile(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
    )
    c_tilelang = kernel(a, b)

    # ---- Reference (cuBLAS) ----
    c_ref = a @ b

    # ---- Correctness ----
    try:
        torch.testing.assert_close(c_tilelang, c_ref, rtol=1e-2, atol=1e-2)
        print("\n[PASS] TileLang GEMM matches PyTorch (rtol=1e-2)")
    except AssertionError as e:
        max_err = (c_tilelang.float() - c_ref.float()).abs().max().item()
        print(f"\n[FAIL] Mismatch! Max absolute error: {max_err:.6f}")
        print(e)

    # ---- Performance ----
    profiler = kernel.get_profiler()
    latency_tl = profiler.do_bench(warmup=25, rep=100)
    latency_pt = do_bench(lambda: a @ b, warmup=25, rep=100)

    # TFLOPS: 2 * M * N * K FLOPs per matmul
    total_flops = 2 * M * N * K
    tflops_tl = total_flops / (latency_tl / 1000) / 1e12
    tflops_pt = total_flops / (latency_pt / 1000) / 1e12

    print("\n--- Performance ---")
    print(f"TileLang GEMM:  {latency_tl:.4f} ms  ({tflops_tl:.2f} TFLOPS)")
    print(f"PyTorch GEMM:   {latency_pt:.4f} ms  ({tflops_pt:.2f} TFLOPS)")
    print(f"Speedup vs PT:  {latency_pt / latency_tl:.2f}x")
    print(f"Gap:            {(latency_tl / latency_pt - 1) * 100:.1f}% slower than PyTorch")

    # ---- CUDA source ----
    src_path = "/tmp/bench_gemm.cu"
    with open(src_path, "w") as f:
        f.write(kernel.get_kernel_source())
    print(f"\nGenerated CUDA source saved to: {src_path}")

    return kernel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TileLang GEMM vs PyTorch Benchmark")
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--block_M", type=int, default=128)
    parser.add_argument("--block_N", type=int, default=128)
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
