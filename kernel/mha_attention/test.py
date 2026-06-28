"""
GEMM + SELU fusion scaffold.

The TileLang kernel body is intentionally left as a TODO. This file provides
the surrounding structure: shapes, reference implementation, CLI, and a place
to fill in the fused GEMM + activation kernel.
"""

import argparse

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T


SELU_ALPHA = 1.6732632423543772
SELU_SCALE = 1.0507009873554805


@tilelang.jit
def gemm_selu_fused(
    A,
    B,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
):
    """TODO: implement fused C = selu(A @ B)."""
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_N), dtype)
        B_shared = T.alloc_shared((block_M, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)

        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        
        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


def ref_gemm_selu(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """PyTorch reference for C = selu(A @ B)."""
    return F.selu(A @ B)


def make_inputs(M: int, N: int, K: int, dtype: torch.dtype, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    A = torch.randn((M, K), device=device, dtype=dtype)
    B = torch.randn((K, N), device=device, dtype=dtype)
    return A, B


def main(
    M: int = 1024,
    N: int = 1024,
    K: int = 1024,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    run_kernel: bool = False,
):
    dtype = torch.float16
    device = "cuda"

    print("GEMM + SELU scaffold")
    print(f"M={M}, N={N}, K={K}")
    print(f"block_M={block_M}, block_N={block_N}, block_K={block_K}")

    A, B = make_inputs(M, N, K, dtype=dtype, device=device)
    ref = ref_gemm_selu(A, B)
    print(f"Reference output shape: {tuple(ref.shape)}")

    # if not run_kernel:
    #     print("TileLang kernel body is blank. Fill gemm_selu_fused before using --run-kernel.")
    #     return None

    kernel = gemm_selu_fused.compile(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
    )
    out = kernel(A, B)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    print("[PASS] TileLang GEMM + SELU matches PyTorch reference.")
    return kernel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GEMM + SELU fusion scaffold")
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--block_M", type=int, default=128)
    parser.add_argument("--block_N", type=int, default=128)
    parser.add_argument("--block_K", type=int, default=32)
    parser.add_argument("--run-kernel", action="store_true", default=False)
    args = parser.parse_args()

    main(
        M=args.M,
        N=args.N,
        K=args.K,
        block_M=args.block_M,
        block_N=args.block_N,
        block_K=args.block_K,
        run_kernel=args.run_kernel,
    )
