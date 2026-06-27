"""
GEMM + Activation Fusion 基准测试。

设计要点：
  TileLang Fused 和 Non-Fused 使用相同的 GEMM 实现 (gemm_only kernel)，
  唯一区别在于 activation 是否在 fragment（寄存器）上就地完成，
  以此隔离 activation 融合的性能影响。

GELU 公式：
  精确版本: gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
  tanh近似: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
"""

from pathlib import Path

import tilelang
import tilelang.language as T
import torch

BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32

SIZES = [512, 1024, 2048, 4096]


@tilelang.jit
def gemm_relu_fused(
    A, B,
    block_M: int, block_N: int, block_K: int
):
    """GEMM + ReLU 融合 kernel。在 fragment 上就地施加 ReLU 后再写回 global memory。"""
    M, N, K = T.const("M, N, K")
    dtype = T.float16
    accum_dtype = T.float32

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)

        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, ko * block_K], A_shared)
            T.copy(B[ko * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        for i, j in T.Parallel(block_M, block_N):
            C_local[i, j] = T.max(C_local[i, j], 0)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


@tilelang.jit
def gemm_gelu_approx_fused(
    A, B,
    block_M: int, block_N: int, block_K: int
):
    """GEMM + GELU (tanh近似) 融合 kernel。"""
    M, N, K = T.const("M, N, K")
    dtype = T.float16
    accum_dtype = T.float32

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)

        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, ko * block_K], A_shared)
            T.copy(B[ko * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        for i, j in T.Parallel(block_M, block_N):
            x = C_local[i, j]
            x_cube = x * x * x
            inner = T.sqrt(T.cast(2.0, accum_dtype) / T.cast(3.141592653589793, accum_dtype)) * (
                x + T.cast(0.044715, accum_dtype) * x_cube
            )
            gelu_out = T.cast(0.5, accum_dtype) * x * (T.cast(1.0, accum_dtype) + T.tanh(inner))
            C_local[i, j] = gelu_out

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


@tilelang.jit
def gemm_gelu_exact_fused(
    A, B,
    block_M: int, block_N: int, block_K: int
):
    """GEMM + GELU (erf精确) 融合 kernel。"""
    M, N, K = T.const("M, N, K")
    dtype = T.float16
    accum_dtype = T.float32

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)

        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, ko * block_K], A_shared)
            T.copy(B[ko * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        for i, j in T.Parallel(block_M, block_N):
            x = C_local[i, j]
            normalized = x / T.sqrt(T.cast(2.0, accum_dtype))
            gelu_out = T.cast(0.5, accum_dtype) * x * (T.cast(1.0, accum_dtype) + T.erf(normalized))
            C_local[i, j] = gelu_out

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


@tilelang.jit
def gemm_only(
    A, B,
    block_M: int, block_N: int, block_K: int
):
    """纯 GEMM kernel，作为非融合 baseline 的矩阵乘法部分。"""
    M, N, K = T.const("M, N, K")
    dtype = T.float16
    accum_dtype = T.float32

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)

        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, ko * block_K], A_shared)
            T.copy(B[ko * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _triton_gemm_relu_kernel(
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        A = A_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
        B = B_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            a_mask = (rm[:, None] < M) & ((k + rk)[None, :] < K)
            b_mask = ((k + rk)[:, None] < K) & (rn[None, :] < N)

            a = tl.load(A, mask=a_mask, other=0.0)
            b = tl.load(B, mask=b_mask, other=0.0)

            acc += tl.dot(a, b)

            A += BLOCK_K * stride_ak
            B += BLOCK_K * stride_bk

        acc = tl.maximum(acc, 0)

        C = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(C, acc.to(tl.float16), mask=c_mask)

    def run_triton_gemm_relu(a, b):
        M, K = a.shape
        _, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _triton_gemm_relu_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return c

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("[警告] Triton 未安装，将跳过 Triton baseline 测试")


def verify_correctness():
    """验证所有 TileLang kernel 的输出与 PyTorch 参考结果一致。"""
    M, N, K = 1024, 1024, 1024

    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    print("=" * 70)
    print("正确性验证")
    print("=" * 70)

    print("\n[1/4] 验证 GEMM + ReLU 融合...")
    kernel_relu = gemm_relu_fused.compile(
        M=M, N=N, K=K,
        block_M=BLOCK_M, block_N=BLOCK_N, block_K=BLOCK_K,
    )
    c_relu = kernel_relu(a, b)
    ref_relu = torch.relu(a @ b)
    torch.testing.assert_close(c_relu, ref_relu, rtol=1e-2, atol=1e-2)
    print("  ✓ GEMM + ReLU 融合: 输出与 PyTorch 一致")

    print("\n[2/4] 验证 GEMM + GELU (tanh近似) 融合...")
    kernel_gelu_approx = gemm_gelu_approx_fused.compile(
        M=M, N=N, K=K,
        block_M=BLOCK_M, block_N=BLOCK_N, block_K=BLOCK_K,
    )
    c_gelu_approx = kernel_gelu_approx(a, b)
    ref_gelu = torch.nn.functional.gelu(a @ b)
    torch.testing.assert_close(c_gelu_approx, ref_gelu, rtol=1e-2, atol=1e-2)
    print("  ✓ GEMM + GELU (tanh近似) 融合: 输出与 PyTorch 一致")

    print("\n[3/4] 验证 GEMM + GELU (erf精确) 融合...")
    kernel_gelu_exact = gemm_gelu_exact_fused.compile(
        M=M, N=N, K=K,
        block_M=BLOCK_M, block_N=BLOCK_N, block_K=BLOCK_K,
    )
    c_gelu_exact = kernel_gelu_exact(a, b)
    torch.testing.assert_close(c_gelu_exact, ref_gelu, rtol=1e-2, atol=1e-2)
    print("  ✓ GEMM + GELU (erf精确) 融合: 输出与 PyTorch 一致")

    print("\n[4/4] 验证纯 GEMM...")
    kernel_gemm = gemm_only.compile(
        M=M, N=N, K=K,
        block_M=BLOCK_M, block_N=BLOCK_N, block_K=BLOCK_K,
    )
    c_gemm = kernel_gemm(a, b)
    ref_gemm = a @ b
    torch.testing.assert_close(c_gemm, ref_gemm, rtol=1e-2, atol=1e-2)
    print("  ✓ 纯 GEMM: 输出与 PyTorch 一致")

    print("\n✅ 所有正确性验证通过！")


def benchmark_kernel(kernel_fn, a, b, warmup=25, rep=100):
    """使用 TileLang profiler 测量 kernel 延迟。"""
    profiler = kernel_fn.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
    latency = profiler.do_bench(input_tensors=[a, b], warmup=warmup, rep=rep)
    return latency


def benchmark_torch(fn, warmup=10, rep=100):
    """使用 CUDA event 精确测量 PyTorch 操作的延迟。"""
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    timings = []
    for _ in range(rep):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))

    return sum(timings) / len(timings)


def benchmark_triton(kernel_fn, a, b, warmup=10, rep=100):
    """使用 Triton 的 benchmarking 工具测量延迟。"""
    import triton

    def wrapper():
        return kernel_fn(a, b)

    latency = triton.testing.do_bench(wrapper, warmup=warmup, rep=rep)
    return latency


def benchmark_nonfused(gemm_kernel, activation_fn, a, b, warmup=10, rep=100):
    """测量非融合路径的延迟：TileLang gemm_only kernel + PyTorch activation。"""
    for _ in range(warmup):
        c = gemm_kernel(a, b)
        activation_fn(c)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    timings = []
    for _ in range(rep):
        start.record()
        c = gemm_kernel(a, b)
        activation_fn(c)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))

    return sum(timings) / len(timings)


def run_all_benchmarks():
    """
    对每种矩阵大小测试所有后端的延迟：
      - TileLang Fused (ReLU / GELU tanh / GELU erf)
      - TileLang Non-Fused (gemm_only kernel + PyTorch activation)
      - cuBLAS (PyTorch mm + activation)
      - Triton (GEMM + Activation 融合)

    TileLang Fused 和 Non-Fused 共享同一 GEMM 实现，仅 activation 位置不同。
    """
    results = {}

    print("编译 TileLang kernels...")
    compiled_kernels = {}
    for size in SIZES:
        compiled_kernels[size] = {
            "gemm_relu": gemm_relu_fused.compile(
                M=size, N=size, K=size,
                block_M=BLOCK_M, block_N=BLOCK_N, block_K=BLOCK_K,
            ),
            "gemm_gelu_approx": gemm_gelu_approx_fused.compile(
                M=size, N=size, K=size,
                block_M=BLOCK_M, block_N=BLOCK_N, block_K=BLOCK_K,
            ),
            "gemm_gelu_exact": gemm_gelu_exact_fused.compile(
                M=size, N=size, K=size,
                block_M=BLOCK_M, block_N=BLOCK_N, block_K=BLOCK_K,
            ),
            "gemm_only": gemm_only.compile(
                M=size, N=size, K=size,
                block_M=BLOCK_M, block_N=BLOCK_N, block_K=BLOCK_K,
            ),
        }
    print("编译完成.\n")

    for size in SIZES:
        print(f"\n{'='*70}")
        print(f"矩阵大小: M=N=K={size}")
        print(f"{'='*70}")

        a = torch.randn(size, size, device="cuda", dtype=torch.float16)
        b = torch.randn(size, size, device="cuda", dtype=torch.float16)

        kernels = compiled_kernels[size]

        print(f"\n  --- GEMM + ReLU (size={size}) ---")

        lat = benchmark_kernel(kernels["gemm_relu"], a, b)
        results[("TileLang Fused", "ReLU", size)] = lat
        print(f"  TileLang Fused (ReLU):          {lat:.4f} ms")

        lat_nf = benchmark_nonfused(kernels["gemm_only"], torch.relu, a, b)
        results[("TileLang Non-Fused", "ReLU", size)] = lat_nf
        print(f"  TileLang Non-Fused (ReLU):      {lat_nf:.4f} ms")

        def cublas_relu():
            return torch.relu(a @ b)
        lat_cublas_relu = benchmark_torch(lambda: cublas_relu())
        results[("cuBLAS", "ReLU", size)] = lat_cublas_relu
        print(f"  cuBLAS + relu:                  {lat_cublas_relu:.4f} ms")

        if HAS_TRITON:
            lat_triton_relu = benchmark_triton(
                lambda x, y: run_triton_gemm_relu(x, y), a, b
            )
            results[("Triton", "ReLU", size)] = lat_triton_relu
            print(f"  Triton (GEMM+ReLU):             {lat_triton_relu:.4f} ms")

        print(f"\n  --- GEMM + GELU (size={size}) ---")

        lat = benchmark_kernel(kernels["gemm_gelu_approx"], a, b)
        results[("TileLang Fused", "GELU (tanh)", size)] = lat
        print(f"  TileLang Fused (GELU tanh):      {lat:.4f} ms")

        lat = benchmark_kernel(kernels["gemm_gelu_exact"], a, b)
        results[("TileLang Fused", "GELU (erf)", size)] = lat
        print(f"  TileLang Fused (GELU erf):       {lat:.4f} ms")

        lat_nf_gelu = benchmark_nonfused(kernels["gemm_only"], torch.nn.functional.gelu, a, b)
        results[("TileLang Non-Fused", "GELU", size)] = lat_nf_gelu
        print(f"  TileLang Non-Fused (GELU):      {lat_nf_gelu:.4f} ms")

        def cublas_gelu():
            return torch.nn.functional.gelu(a @ b)
        lat_cublas_gelu = benchmark_torch(lambda: cublas_gelu())
        results[("cuBLAS", "GELU", size)] = lat_cublas_gelu
        print(f"  cuBLAS + gelu:                  {lat_cublas_gelu:.4f} ms")

    return results


def analyze_results(results):
    """分析性能数据：融合加速比、GELU 变体对比、与 cuBLAS 对比、汇总表。"""
    print("\n" + "=" * 70)
    print("性能分析")
    print("=" * 70)

    print("\n--- 融合加速比 (vs TileLang Non-Fused) ---")
    print(f"{'Size':<10} {'ReLU Speedup':<15} {'GELU Speedup':<15}")
    print("-" * 40)
    for size in SIZES:
        relu_fused = results.get(("TileLang Fused", "ReLU", size), 0)
        relu_nonfused = results.get(("TileLang Non-Fused", "ReLU", size), 0)
        gelu_fused = results.get(("TileLang Fused", "GELU (tanh)", size), 0)
        gelu_nonfused = results.get(("TileLang Non-Fused", "GELU", size), 0)

        relu_speedup = relu_nonfused / relu_fused if relu_fused > 0 else 0
        gelu_speedup = gelu_nonfused / gelu_fused if gelu_fused > 0 else 0
        print(f"{size:<10} {relu_speedup:<15.2f}x {gelu_speedup:<15.2f}x")

    print("\n--- GELU 变体性能对比 (tanh vs erf) ---")
    print(f"{'Size':<10} {'tanh (ms)':<15} {'erf (ms)':<15} {'Ratio (erf/tanh)':<15}")
    print("-" * 55)
    for size in SIZES:
        tanh_lat = results.get(("TileLang Fused", "GELU (tanh)", size), 0)
        erf_lat = results.get(("TileLang Fused", "GELU (erf)", size), 0)
        ratio = erf_lat / tanh_lat if tanh_lat > 0 else 0
        print(f"{size:<10} {tanh_lat:<15.4f} {erf_lat:<15.4f} {ratio:<15.2f}")

    print("\n--- 与 cuBLAS 性能对比 ---")
    print(f"{'Size':<10} {'ReLU (TL/cuBLAS)':<20} {'GELU (TL/cuBLAS)':<20}")
    print("-" * 50)
    for size in SIZES:
        relu_tl = results.get(("TileLang Fused", "ReLU", size), 0)
        relu_cublas = results.get(("cuBLAS", "ReLU", size), 0)
        gelu_tl = results.get(("TileLang Fused", "GELU (tanh)", size), 0)
        gelu_cublas = results.get(("cuBLAS", "GELU", size), 0)

        relu_ratio = relu_tl / relu_cublas if relu_cublas > 0 else 0
        gelu_ratio = gelu_tl / gelu_cublas if gelu_cublas > 0 else 0
        print(f"{size:<10} {relu_ratio:<20.2f} {gelu_ratio:<20.2f}")

    print("\n--- 性能汇总表 (单位: ms) ---")
    print()
    header = f"| {'Size':<8} | {'TL Fused ReLU':<14} | {'TL NF ReLU':<11} | {'cuBLAS ReLU':<12} | {'TL Fused GELU':<14} | {'TL NF GELU':<11} | {'cuBLAS GELU':<12} |"
    sep = "|" + "-" * 10 + "|" + "-" * 16 + "|" + "-" * 13 + "|" + "-" * 14 + "|" + "-" * 16 + "|" + "-" * 13 + "|" + "-" * 14 + "|"
    print(header)
    print(sep)

    for size in SIZES:
        relu_f = results.get(("TileLang Fused", "ReLU", size), 0)
        relu_nf = results.get(("TileLang Non-Fused", "ReLU", size), 0)
        relu_cublas = results.get(("cuBLAS", "ReLU", size), 0)
        gelu_f = results.get(("TileLang Fused", "GELU (tanh)", size), 0)
        gelu_nf = results.get(("TileLang Non-Fused", "GELU", size), 0)
        gelu_cublas = results.get(("cuBLAS", "GELU", size), 0)

        print(f"| {size:<8} | {relu_f:<14.4f} | {relu_nf:<11.4f} | {relu_cublas:<12.4f} | {gelu_f:<14.4f} | {gelu_nf:<11.4f} | {gelu_cublas:<12.4f} |")
<<<<<<< HEAD


def plot_results(results, output_path=None):
    """使用 matplotlib 生成 benchmark 结果图。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[警告] matplotlib 未安装，跳过绘图。可运行 `pip install matplotlib` 后重试。")
        return None

    output_path = Path(output_path or Path(__file__).with_name("gemm_activation_fusion_results.png"))

    sizes = SIZES
    x = list(range(len(sizes)))

    relu_series = [
        ("TileLang Fused", [results.get(("TileLang Fused", "ReLU", size), 0) for size in sizes]),
        ("TileLang Non-Fused", [results.get(("TileLang Non-Fused", "ReLU", size), 0) for size in sizes]),
        ("cuBLAS", [results.get(("cuBLAS", "ReLU", size), 0) for size in sizes]),
    ]
    if any(("Triton", "ReLU", size) in results for size in sizes):
        relu_series.append(("Triton", [results.get(("Triton", "ReLU", size), 0) for size in sizes]))

    gelu_series = [
        ("TileLang Fused (tanh)", [results.get(("TileLang Fused", "GELU (tanh)", size), 0) for size in sizes]),
        ("TileLang Fused (erf)", [results.get(("TileLang Fused", "GELU (erf)", size), 0) for size in sizes]),
        ("TileLang Non-Fused", [results.get(("TileLang Non-Fused", "GELU", size), 0) for size in sizes]),
        ("cuBLAS", [results.get(("cuBLAS", "GELU", size), 0) for size in sizes]),
    ]

    relu_speedups = []
    gelu_speedups = []
    for size in sizes:
        relu_fused = results.get(("TileLang Fused", "ReLU", size), 0)
        relu_nonfused = results.get(("TileLang Non-Fused", "ReLU", size), 0)
        gelu_fused = results.get(("TileLang Fused", "GELU (tanh)", size), 0)
        gelu_nonfused = results.get(("TileLang Non-Fused", "GELU", size), 0)
        relu_speedups.append(relu_nonfused / relu_fused if relu_fused > 0 else 0)
        gelu_speedups.append(gelu_nonfused / gelu_fused if gelu_fused > 0 else 0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("GEMM + Activation Fusion Benchmark", fontsize=16, fontweight="bold")

    ax = axes[0][0]
    width = 0.8 / len(relu_series)
    for idx, (label, values) in enumerate(relu_series):
        offset = (idx - (len(relu_series) - 1) / 2) * width
        ax.bar([pos + offset for pos in x], values, width=width, label=label)
    ax.set_title("GEMM + ReLU Latency")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(size) for size in sizes])
    ax.set_xlabel("M=N=K")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0][1]
    width = 0.8 / len(gelu_series)
    for idx, (label, values) in enumerate(gelu_series):
        offset = (idx - (len(gelu_series) - 1) / 2) * width
        ax.bar([pos + offset for pos in x], values, width=width, label=label)
    ax.set_title("GEMM + GELU Latency")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(size) for size in sizes])
    ax.set_xlabel("M=N=K")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1][0]
    ax.plot(x, relu_speedups, marker="o", linewidth=2, label="ReLU")
    ax.plot(x, gelu_speedups, marker="s", linewidth=2, label="GELU")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Fusion Speedup vs TileLang Non-Fused")
    ax.set_ylabel("Speedup (x)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(size) for size in sizes])
    ax.set_xlabel("M=N=K")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1][1]
    gelu_tanh = [results.get(("TileLang Fused", "GELU (tanh)", size), 0) for size in sizes]
    gelu_erf = [results.get(("TileLang Fused", "GELU (erf)", size), 0) for size in sizes]
    ax.plot(x, gelu_tanh, marker="o", linewidth=2, label="tanh approx")
    ax.plot(x, gelu_erf, marker="s", linewidth=2, label="erf exact")
    ax.set_title("TileLang Fused GELU Variants")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(size) for size in sizes])
    ax.set_xlabel("M=N=K")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\n性能图已保存到: {output_path}")
    return output_path
=======
>>>>>>> gemm_activation


def main():
    if not torch.cuda.is_available():
        print("错误: 需要 CUDA GPU 才能运行此脚本")
        return

    gpu_name = torch.cuda.get_device_properties(0).name
    print(f"GPU: {gpu_name}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print(f"TileLang: {tilelang.__version__}")
    print(f"Triton 可用: {HAS_TRITON}")

    verify_correctness()

    print("\n\n" + "=" * 70)
    print("性能基准测试")
    print("=" * 70)
    results = run_all_benchmarks()

    analyze_results(results)
    plot_results(results)

    print("\n✅ 所有任务完成！")


if __name__ == "__main__":
    main()
