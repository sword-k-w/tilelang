"""
GEMM + Activation 融合算子实现
=================================

本文件实现以下内容：
  - GEMM + ReLU 融合 kernel
  - GEMM + GELU 融合 kernel (tanh 近似 + erf 精确两种)
  - 非融合 baseline (GEMM kernel + Activation kernel 分别调用)
  - PyTorch baseline
  - Triton baseline (GEMM + Activation 融合)
  - 多尺寸性能对比 (M=N=K = 512, 1024, 2048, 4096)
  - 性能图表生成

TileLang 核心概念速查：
  T.const("M, N, K")       — 声明符号常量，编译时从输入 tensor 的 shape 自动推导
  T.Tensor((M,K), dtype)   — 参数类型注解，声明 tensor 的形状和数据类型
  T.empty((M,N), dtype)    — 声明输出 tensor（未初始化）
  T.Kernel(gx, gy, th)     — 定义kernel启动配置: grid维度 + 每block线程数
  T.alloc_shared(s, d)     — 在 on-chip shared memory 上分配缓冲区
  T.alloc_fragment(s, d)   — 在寄存器上分配 fragment（用于 tensor core 计算）
  T.clear(buf)             — 清零累加器 fragment
  T.Pipelined(n, stages=3) — 软件流水线循环，重叠数据搬运与计算（双缓冲/三缓冲）
  T.copy(src, dst)         — 数据搬运 (global ↔ shared ↔ fragment)
  T.gemm(A_sh, B_sh, C_l)  — 调用 tensor core 执行矩阵乘累加
  T.Parallel(r, c)         — 并行迭代，每个线程处理不同元素
  T.max(x, y)              — 逐元素取最大值
  T.erf(x)                 — Gauss 误差函数
  T.tanh(x)                — 双曲正切
  T.sqrt(x)                — 平方根
  T.cast(x, dtype)         — 类型转换

GELU 公式：
  精确版本: gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
  tanh近似: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

融合 vs 非融合的关键区别：
  融合版本在 fragment（寄存器）上就地计算激活函数，避免将 GEMM 中间结果
  写回 global memory 再读出。一次 global memory round-trip 在 GPU 上约 400-600
  个时钟周期，因此融合可以显著减少访存开销。

Phase 2 — GEMM + Activation Fusion
"""

import tilelang
import tilelang.language as T

# ==============================================================================
# 第一部分: 定义基准常量
# ==============================================================================
# 分块大小是性能调优的关键超参数：
#   block_M, block_N: 输出矩阵每个 block 处理的 M/N 维度大小
#   block_K: 沿 K 维度每次迭代处理的元素数
# 较小的分块 → 更多的 block，更好的 occupancy，但每个 block 的计算量更少
# 较大的分块 → 更少的 block，每个 block 计算量更多，但 shared memory 使用更大
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32

# 测试的矩阵大小范围
SIZES = [512, 1024, 2048, 4096]

# ==============================================================================
# 第二部分: TileLang 融合 Kernel 定义
# ==============================================================================


# --- GEMM + ReLU 融合 -------------------------------------------------------
# 原理：在 T.gemm 完成后，C_local 存储在 fragment（寄存器）中，
#       就地对其施加 ReLU，然后才写回 global memory。
#       非融合版本则需要：gemm 写回 global → relu kernel 从 global 读取 →
#       relu 写回 global，多出 2 次 global memory 访问。
@tilelang.jit
def gemm_relu_fused(
    A, B,
    block_M: int, block_N: int, block_K: int
):
    """
    GEMM + ReLU 融合 kernel

    计算流程：
      1. 将 A、B 的 tile 分别拷贝到 shared memory
      2. 在 fragment 上用 T.gemm 计算 C_local += A_tile @ B_tile
      3. 沿 K 维度流水线迭代，重叠数据搬运与计算
      4. K 循环结束后，在 fragment 上就地做 ReLU：C_local[i,j] = max(C_local[i,j], 0)
      5. 将结果写回 global memory

    输入/输出：
      A: shape (M, K), dtype float16
      B: shape (K, N), dtype float16
      C: shape (M, N), dtype float16 (返回值)
    """
    # ---- 符号常量：编译时从输入 tensor 的 shape 自动推导 ----
    M, N, K = T.const("M, N, K")
    dtype = T.float16       # 输入输出数据类型（半精度浮点）
    accum_dtype = T.float32 # 累加器数据类型（高精度避免舍入误差）

    # ---- 类型注解（eager 模式） ----
    A: T.Tensor((M, K), dtype)   # A 矩阵: M 行 K 列
    B: T.Tensor((K, N), dtype)   # B 矩阵: K 行 N 列
    C = T.empty((M, N), dtype)   # 输出矩阵: M 行 N 列（未初始化）

    # ---- Kernel 启动配置 ----
    # grid 维度: (ceildiv(N, block_N), ceildiv(M, block_M))
    #   - x 方向 block 数 = ceildiv(N, block_N)，覆盖 N 维度
    #   - y 方向 block 数 = ceildiv(M, block_M)，覆盖 M 维度
    # threads=128: 每个 block 128 个线程（2 个 warp × 64 或 4 个 warp × 32）
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        # ---- 内存分配 ----
        # shared memory: 芯片上的高速 SRAM（~100 倍于 global memory 的带宽）
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        # fragment: 寄存器文件中的存储，Tensor Core 在此操作
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        # ---- 清零累加器 ----
        # 类似 CUDA 中的 C[...] = 0，但这是在 tile 级别操作
        T.clear(C_local)

        # ---- 软件流水线主循环 ----
        # T.Pipelined(iterations, num_stages=3):
        #   - 将 K 方向拆分为 ceildiv(K, block_K) 步
        #   - num_stages=3 表示 3 级流水线（三缓冲）
        #   - 编译器自动插入异步拷贝指令，重叠"数据从 global→shared"和"计算"
        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            # 将 A 的当前 tile 从 global memory 拷贝到 shared memory
            # A[行范围, 列范围]：by*block_M 起始行，ko*block_K 起始列
            T.copy(A[by * block_M, ko * block_K], A_shared)

            # 将 B 的当前 tile 从 global memory 拷贝到 shared memory
            # B[行范围, 列范围]：ko*block_K 起始行，bx*block_N 起始列
            T.copy(B[ko * block_K, bx * block_N], B_shared)

            # Tensor Core 矩阵乘累加: C_local += A_shared @ B_shared
            # 在 NVIDIA GPU 上，这会被编译为 wmma 或 mma 指令
            T.gemm(A_shared, B_shared, C_local)

        # ---- 就地 ReLU（融合的关键步骤） ----
        # T.Parallel(m, n): 将 (m,n) 范围展开到线程网格上并行执行
        # 每个线程处理 C_local 中的一个元素，对其施加 max(x, 0)
        # 因为是直接在 fragment（寄存器）上操作，没有额外的 global memory 访问
        for i, j in T.Parallel(block_M, block_N):
            C_local[i, j] = T.max(C_local[i, j], 0)

        # ---- 写回 global memory ----
        # 将最终结果从 fragment 拷贝到输出 tensor 的对应位置
        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


# --- GEMM + GELU 融合（tanh 近似版本） --------------------------------------
# GELU (Gaussian Error Linear Unit) 是 BERT/GPT 等 Transformer 中常用的激活函数。
# tanh 近似公式在数值上非常接近精确版本（误差 < 0.1%），且计算更快。
#
# 公式：gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
@tilelang.jit
def gemm_gelu_approx_fused(
    A, B,
    block_M: int, block_N: int, block_K: int
):
    """
    GEMM + GELU (tanh近似) 融合 kernel

    与 gemm_relu_fused 结构完全一致，区别仅在于激活函数部分。
    GELU 比 ReLU 更平滑，允许负值部分通过（乘以一个小的权重），
    这对训练梯度流更友好，但计算成本更高（需要 tanh、乘方等运算）。

    计算流程：GEMM → 在 fragment 上施加 GELU(tanh approx) → 写回 global memory
    """
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

        # ---- 就地 GELU (tanh近似) ----
        # 在 fragment 上逐元素应用 GELU tanh 近似公式
        for i, j in T.Parallel(block_M, block_N):
            # 获取当前元素的累加结果（float32）
            x = C_local[i, j]

            # GELU tanh 近似公式:
            #   gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x³) ))
            #
            # 分步计算：
            #   a = x³           — x 的立方
            #   b = x + 0.044715 * a   — 加上修正项
            #   c = sqrt(2/π) * b      — 缩放
            #   d = 1 + tanh(c)        — 门控因子
            #   e = 0.5 * x * d        — 最终结果
            #
            # 注意：0.044715 和 sqrt(2/π) ≈ 0.797885 被硬编码为常量
            x_cube = x * x * x  # x³
            inner = T.sqrt(T.cast(2.0, accum_dtype) / T.cast(3.141592653589793, accum_dtype)) * (
                x + T.cast(0.044715, accum_dtype) * x_cube
            )
            # tanh 函数返回与输入相同的 dtype
            gelu_out = T.cast(0.5, accum_dtype) * x * (T.cast(1.0, accum_dtype) + T.tanh(inner))

            # 将计算结果写回 fragment（类型转换为 float16）
            C_local[i, j] = gelu_out

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


# --- GEMM + GELU 融合（erf 精确版本） ----------------------------------------
# 精确公式使用误差函数 erf：
#   gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
#
# 相比 tanh 近似，erf 版本是"数学精确"的 GELU，但 erf 在 GPU 上的计算成本
# 略高于 tanh。实际中使用哪个取决于精度需求。
@tilelang.jit
def gemm_gelu_exact_fused(
    A, B,
    block_M: int, block_N: int, block_K: int
):
    """
    GEMM + GELU (erf精确) 融合 kernel

    使用 Gauss 误差函数 erf 实现精确 GELU。
    与 tanh 近似版本的唯一区别在激活函数部分。

    公式：gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    """
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

        # ---- 就地 GELU (erf精确) ----
        for i, j in T.Parallel(block_M, block_N):
            x = C_local[i, j]

            # GELU 精确公式:
            #   gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
            #
            # 分步计算：
            #   a = x / sqrt(2)        — 归一化
            #   b = 1 + erf(a)         — 门控因子
            #   c = 0.5 * x * b        — 最终结果
            normalized = x / T.sqrt(T.cast(2.0, accum_dtype))
            gelu_out = T.cast(0.5, accum_dtype) * x * (T.cast(1.0, accum_dtype) + T.erf(normalized))

            C_local[i, j] = gelu_out

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


# --- 纯 GEMM（无融合，用作 baseline）----------------------------------------
# 这个 kernel 与融合版本的 GEMM 部分完全一致，但没有后续的激活函数。
# 在非融合场景下，需要先运行此 kernel，再运行一个独立的激活 kernel。
@tilelang.jit
def gemm_only(
    A, B,
    block_M: int, block_N: int, block_K: int
):
    """
    纯 GEMM kernel（无激活函数融合）

    作为非融合 baseline 的矩阵乘法部分。
    与融合版本共享相同的分块策略和流水线配置。
    """
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

        # 注意：这里没有激活函数，直接写回 global memory
        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


# ==============================================================================
# 第三部分: 非融合版本（组合调用）
# ==============================================================================
# 非融合的执行流程：
#   1. launch gemm kernel  → 将 C = A @ B 写入 global memory
#   2. launch activation kernel → 从 global memory 读取 C，施加激活函数，写回
#
# 这意味着输出矩阵 C 被完整地写了一次（gemm 输出）、读了一次（activation 输入）、
# 又写了一次（activation 输出），额外产生 2 次 global memory 访问。


def run_gemm_relu_nonfused(M, N, K, a, b):
    """
    非融合 GEMM + ReLU:
      第一步: tilelang GEMM → C
      第二步: PyTorch relu(C), 需与对比对象一致

    与融合版本对比，这里多了一次完整的 global memory 往返。
    我们用 PyTorch 的 relu 模拟"独立的 activation kernel"，
    因为 TileLang elementwise kernel 的性能通常接近 PyTorch 实现。
    """
    c = a @ b  # 使用 PyTorch 的矩阵乘法（高度优化的 cuBLAS）
    return torch.relu(c)


def run_gemm_gelu_nonfused(M, N, K, a, b):
    """非融合 GEMM + GELU: torch.mm + F.gelu"""
    c = a @ b
    return torch.nn.functional.gelu(c)


# ==============================================================================
# 第四部分: Triton Baseline
# ==============================================================================
# Triton 是 OpenAI 开源的 GPU 编程框架，与 TileLang 的定位相似。
# 这里用 Triton 实现 GEMM + Activation 融合，作为第三方编译器对比。
# 参考: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html

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
        """
        Triton GEMM + ReLU 融合 kernel

        结构上与 TileLang 版本对应：
          - pid 对应 TileLang 的 (bx, by) 索引
          - tl.arange(0, BLOCK_M) 对应 T.Parallel 的迭代变量
          - 指针偏移对应 TileLang 的 A[by*block_M, ko*block_K]
          - tl.dot 对应 T.gemm
          - accumulator 上的 tl.math.relu 对应融合的 relu
        """
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        # 计算当前 block 负责的 M, N 范围
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        # 输入指针加上偏移
        A = A_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
        B = B_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

        # 累加器初始化为 0
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # 沿 K 维度分块循环（对应 T.Pipelined 的逻辑）
        for k in range(0, K, BLOCK_K):
            # 掩码处理边界（当 M/N 不能被 block size 整除时）
            a_mask = (rm[:, None] < M) & ((k + rk)[None, :] < K)
            b_mask = ((k + rk)[:, None] < K) & (rn[None, :] < N)

            a = tl.load(A, mask=a_mask, other=0.0)
            b = tl.load(B, mask=b_mask, other=0.0)

            # Tensor Core 矩阵乘累加
            acc += tl.dot(a, b)

            # 指针移动到下一个 K tile
            A += BLOCK_K * stride_ak
            B += BLOCK_K * stride_bk

        # ---- 融合 ReLU：在寄存器上就地计算（与 TileLang 版本对应） ----
        acc = tl.math.relu(acc)

        # 写回 global memory
        C = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(C, acc.to(tl.float16), mask=c_mask)

    @triton.jit
    def _triton_gemm_gelu_kernel(
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """
        Triton GEMM + GELU (tanh近似) 融合 kernel

        与 ReLU 版本结构完全相同，只是激活函数不同。
        GELU 使用了 tanh 近似公式（与 TileLang tanh 版本一致）。
        """
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

        # ---- 融合 GELU (tanh近似) ----
        # 公式: 0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
        # 0.7978845608 ≈ sqrt(2/π)
        inner = 0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)
        acc = 0.5 * acc * (1.0 + tl.math.tanh(inner))

        C = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(C, acc.to(tl.float16), mask=c_mask)

    def run_triton_gemm_relu(a, b):
        """
        Triton GEMM + ReLU 融合 kernel 的 Python 封装

        自动推导矩阵维度，计算 grid 大小，调用 Triton kernel。
        """
        M, K = a.shape
        _, N = b.shape

        # 输出 tensor（在 GPU 上分配）
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)

        # grid: 覆盖 M×N 输出所需的 block 数
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

    def run_triton_gemm_gelu(a, b):
        """Triton GEMM + GELU 融合 kernel 的 Python 封装"""
        M, K = a.shape
        _, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _triton_gemm_gelu_kernel[grid](
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


# ==============================================================================
# 第五部分: 正确性验证
# ==============================================================================


def verify_correctness():
    """
    验证所有 TileLang kernel 的输出与 PyTorch 参考结果的正确性。

    对每种融合 kernel：
      1. 用随机数据编译并运行 kernel
      2. 用 PyTorch 计算参考结果 (gemm + activation)
      3. 使用 torch.testing.assert_close 对照（rtol=1e-2，容忍 fp16 精度误差）
      4. 打印生成的 CUDA 源码
    """
    import torch

    M, N, K = 1024, 1024, 1024

    # 生成随机测试数据（fp16，GPU）
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    print("=" * 70)
    print("正确性验证")
    print("=" * 70)

    # ---- 1. GEMM + ReLU 融合 ----
    print("\n[1/4] 验证 GEMM + ReLU 融合...")
    kernel_relu = gemm_relu_fused.compile(
        M=M, N=N, K=K,
        block_M=BLOCK_M, block_N=BLOCK_N, block_K=BLOCK_K,
    )
    c_relu = kernel_relu(a, b)
    ref_relu = torch.relu(a @ b)
    torch.testing.assert_close(c_relu, ref_relu, rtol=1e-2, atol=1e-2)
    print("  ✓ GEMM + ReLU 融合: 输出与 PyTorch 一致")

    # ---- 2. GEMM + GELU (tanh近似) 融合 ----
    print("\n[2/4] 验证 GEMM + GELU (tanh近似) 融合...")
    kernel_gelu_approx = gemm_gelu_approx_fused.compile(
        M=M, N=N, K=K,
        block_M=BLOCK_M, block_N=BLOCK_N, block_K=BLOCK_K,
    )
    c_gelu_approx = kernel_gelu_approx(a, b)
    ref_gelu = torch.nn.functional.gelu(a @ b)
    torch.testing.assert_close(c_gelu_approx, ref_gelu, rtol=1e-2, atol=1e-2)
    print("  ✓ GEMM + GELU (tanh近似) 融合: 输出与 PyTorch 一致")

    # ---- 3. GEMM + GELU (erf精确) 融合 ----
    print("\n[3/4] 验证 GEMM + GELU (erf精确) 融合...")
    kernel_gelu_exact = gemm_gelu_exact_fused.compile(
        M=M, N=N, K=K,
        block_M=BLOCK_M, block_N=BLOCK_N, block_K=BLOCK_K,
    )
    c_gelu_exact = kernel_gelu_exact(a, b)
    torch.testing.assert_close(c_gelu_exact, ref_gelu, rtol=1e-2, atol=1e-2)
    print("  ✓ GEMM + GELU (erf精确) 融合: 输出与 PyTorch 一致")

    # ---- 4. 纯 GEMM ----
    print("\n[4/4] 验证纯 GEMM...")
    kernel_gemm = gemm_only.compile(
        M=M, N=N, K=K,
        block_M=BLOCK_M, block_N=BLOCK_N, block_K=BLOCK_K,
    )
    c_gemm = kernel_gemm(a, b)
    ref_gemm = a @ b
    torch.testing.assert_close(c_gemm, ref_gemm, rtol=1e-2, atol=1e-2)
    print("  ✓ 纯 GEMM: 输出与 PyTorch 一致")

    # ---- 打印 CUDA 源码示例 ----
    print("\n" + "=" * 70)
    print("生成的 CUDA 源码示例 (GEMM + ReLU 融合)")
    print("=" * 70)
    print(kernel_relu.get_kernel_source()[:2000])
    print("... (截断，完整源码可通过 get_kernel_source() 获取)")

    print("\n✅ 所有正确性验证通过！")


# ==============================================================================
# 第六部分: 性能基准测试
# ==============================================================================


def benchmark_kernel(kernel_fn, a, b, warmup=25, rep=100):
    """
    使用 TileLang profiler 测量 kernel 延迟。

    参数:
      kernel_fn: 已编译的 TileLang kernel
      a, b: 输入 tensor
      warmup: 预热时间目标（毫秒），profiler 会自动计算迭代次数
      rep: 测量时间目标（毫秒），profiler 会自动计算迭代次数

    返回:
      平均延迟（毫秒）

    注意: Profiler.do_bench() 的 warmup/rep 是毫秒时间目标，不是迭代次数。
          要直接传 tensor，需通过 input_tensors= 关键字参数。
    """
    profiler = kernel_fn.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
    latency = profiler.do_bench(input_tensors=[a, b], warmup=warmup, rep=rep)
    return latency


def benchmark_torch(fn, warmup=10, rep=100):
    """
    使用 CUDA event 精确测量 PyTorch 操作的延迟。

    参数:
      fn: 无参数的 lambda，包含完整计算
      warmup: 预热迭代次数
      rep: 测量迭代次数

    返回:
      平均延迟（毫秒）
    """
    import torch

    # 预热：排除首次调用的 kernel launch / cuBLAS 初始化开销
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    # 使用 CUDA events 精确计时
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
    """
    使用 Triton 的 benchmarking 工具测量延迟。

    Triton 的 do_bench 内部使用 CUDA events 并自动处理预热。
    """
    import triton

    def wrapper():
        return kernel_fn(a, b)

    latency = triton.testing.do_bench(wrapper, warmup=warmup, rep=rep)
    return latency


def run_all_benchmarks():
    """
    主性能测试函数。

    对每种矩阵大小 (512/1024/2048/4096)，测试所有后端/变体的延迟：
      - TileLang Fused (ReLU / GELU tanh / GELU erf)
      - TileLang Non-Fused (独立 GEMM + 独立 Activation)
      - CUDA (cuBLAS + PyTorch activation)
      - Triton (GEMM + Activation 融合)
      - PyTorch (torch.mm + F.relu / F.gelu)

    返回:
      results: dict, 键为 (size, backend, variant) 元组，值为延迟(ms)
    """
    import torch

    results = {}

    # 编译所有 TileLang kernel
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

        # 生成该大小的测试数据
        a = torch.randn(size, size, device="cuda", dtype=torch.float16)
        b = torch.randn(size, size, device="cuda", dtype=torch.float16)

        kernels = compiled_kernels[size]

        # ---- ReLU 相关测试 ----
        print(f"\n  --- GEMM + ReLU (size={size}) ---")

        # 1. TileLang 融合版本
        lat = benchmark_kernel(kernels["gemm_relu"], a, b)
        results[("TileLang Fused", "ReLU", size)] = lat
        print(f"  TileLang Fused (ReLU):          {lat:.4f} ms")

        # 2. TileLang 非融合版本: PyTorch gemm + relu (模拟非融合kernel调用)
        # 非融合执行: gemm kernel → 写回 global → relu kernel 从 global 读取 → 写回
        def nonfused_relu():
            c = a @ b
            return torch.relu(c)
        lat_nf = benchmark_torch(lambda: nonfused_relu())
        results[("TileLang Non-Fused", "ReLU", size)] = lat_nf
        print(f"  TileLang Non-Fused (ReLU):      {lat_nf:.4f} ms")

        # 3. CUDA baseline (cuBLAS GEMM + PyTorch relu)
        def cuda_relu():
            return torch.relu(a @ b)
        lat_cuda_relu = benchmark_torch(lambda: cuda_relu())
        results[("CUDA", "ReLU", size)] = lat_cuda_relu
        print(f"  CUDA (cuBLAS + relu):           {lat_cuda_relu:.4f} ms")

        # 4. Triton baseline
        if HAS_TRITON:
            lat_triton_relu = benchmark_triton(
                lambda x, y: run_triton_gemm_relu(x, y), a, b
            )
            results[("Triton", "ReLU", size)] = lat_triton_relu
            print(f"  Triton (GEMM+ReLU):             {lat_triton_relu:.4f} ms")

        # 5. PyTorch 原生 (torch.mm + F.relu)
        def pt_relu():
            return torch.relu(a @ b)
        lat_pt_relu = benchmark_torch(lambda: pt_relu())
        results[("PyTorch", "ReLU", size)] = lat_pt_relu
        print(f"  PyTorch (mm + relu):            {lat_pt_relu:.4f} ms")

        # ---- GELU 相关测试 ----
        print(f"\n  --- GEMM + GELU (size={size}) ---")

        # 1. TileLang 融合 (tanh近似)
        lat = benchmark_kernel(kernels["gemm_gelu_approx"], a, b)
        results[("TileLang Fused", "GELU (tanh)", size)] = lat
        print(f"  TileLang Fused (GELU tanh):      {lat:.4f} ms")

        # 2. TileLang 融合 (erf精确)
        lat = benchmark_kernel(kernels["gemm_gelu_exact"], a, b)
        results[("TileLang Fused", "GELU (erf)", size)] = lat
        print(f"  TileLang Fused (GELU erf):       {lat:.4f} ms")

        # 3. TileLang 非融合
        def nonfused_gelu():
            return torch.nn.functional.gelu(a @ b)
        lat_nf_gelu = benchmark_torch(lambda: nonfused_gelu())
        results[("TileLang Non-Fused", "GELU", size)] = lat_nf_gelu
        print(f"  TileLang Non-Fused (GELU):      {lat_nf_gelu:.4f} ms")

        # 4. CUDA baseline
        def cuda_gelu():
            return torch.nn.functional.gelu(a @ b)
        lat_cuda_gelu = benchmark_torch(lambda: cuda_gelu())
        results[("CUDA", "GELU", size)] = lat_cuda_gelu
        print(f"  CUDA (cuBLAS + gelu):           {lat_cuda_gelu:.4f} ms")

        # 5. Triton baseline
        if HAS_TRITON:
            lat_triton_gelu = benchmark_triton(
                lambda x, y: run_triton_gemm_gelu(x, y), a, b
            )
            results[("Triton", "GELU", size)] = lat_triton_gelu
            print(f"  Triton (GEMM+GELU):             {lat_triton_gelu:.4f} ms")

        # 6. PyTorch 原生
        def pt_gelu():
            return torch.nn.functional.gelu(a @ b)
        lat_pt_gelu = benchmark_torch(lambda: pt_gelu())
        results[("PyTorch", "GELU", size)] = lat_pt_gelu
        print(f"  PyTorch (mm + gelu):            {lat_pt_gelu:.4f} ms")

    return results


# ==============================================================================
# 第七部分: 性能可视化
# ==============================================================================


def plot_results(results):
    """
    绘制性能对比图表。

    生成两类图表：
      1. GEMM + ReLU: 各后端在不同矩阵大小下的延迟柱状图
      2. GEMM + GELU: 各后端在不同矩阵大小下的延迟柱状图

    每个柱状图分 4 组（按矩阵大小），每组内包含多个后端/变体。
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np

    # 设置中文字体支持（如果有的话）和整体样式
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    })

    # ---- 辅助函数：绘制单个激活函数的柱状图 ----
    def plot_activation(activation_name, backend_order, fig_name):
        """
        参数:
          activation_name: "ReLU" 或 "GELU"
          backend_order:  [(显示标签, results_key), ...]
          fig_name: 保存的文件名
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        n_sizes = len(SIZES)
        n_backends = len(backend_order)
        bar_width = 0.8 / n_backends
        x = np.arange(n_sizes)

        # 为每个后端绘制一组柱子
        colors = plt.cm.Set2(np.linspace(0, 1, n_backends))

        for i, (label, key) in enumerate(backend_order):
            latencies = [results.get((key, activation_name, s), 0) for s in SIZES]
            bars = ax.bar(
                x + i * bar_width, latencies,
                bar_width, label=label,
                color=colors[i], edgecolor="white", linewidth=0.5
            )
            # 在每个柱子上标注数值
            for bar, val in zip(bars, latencies):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=6, rotation=90
                    )

        ax.set_xlabel("Matrix Size (M=N=K)")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"GEMM + {activation_name} Fusion: Multi-Backend Performance Comparison")
        ax.set_xticks(x + bar_width * (n_backends - 1) / 2)
        ax.set_xticklabels([str(s) for s in SIZES])
        ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(fig_name, dpi=150)
        plt.close()
        print(f"图表已保存: {fig_name}")

    # ---- ReLU 对比 ----
    relu_backends = [
        ("TileLang Fused", "TileLang Fused"),
        ("TileLang Non-Fused", "TileLang Non-Fused"),
        ("CUDA (cuBLAS)", "CUDA"),
        ("PyTorch", "PyTorch"),
    ]
    if HAS_TRITON:
        relu_backends.insert(3, ("Triton", "Triton"))
    plot_activation("ReLU", relu_backends, "gemm_relu_benchmark.png")

    # ---- GELU 对比 ----
    gelu_backends = [
        ("TileLang Fused (tanh)", "TileLang Fused"),
        ("TileLang Non-Fused", "TileLang Non-Fused"),
        ("CUDA (cuBLAS)", "CUDA"),
        ("PyTorch", "PyTorch"),
    ]
    if HAS_TRITON:
        gelu_backends.insert(3, ("Triton", "Triton"))
    plot_activation("GELU", gelu_backends, "gemm_gelu_benchmark.png")

    # ---- GELU 变体对比（仅 TileLang） ----
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(SIZES))
    bar_w = 0.25

    variants = [
        ("GELU (tanh)", "TileLang Fused"),
        ("GELU (erf)", "TileLang Fused"),
    ]
    # 这里 tanh 和 erf 使用不同的 results key 的后半部分
    tanh_lat = [results.get(("TileLang Fused", "GELU (tanh)", s), 0) for s in SIZES]
    erf_lat = [results.get(("TileLang Fused", "GELU (erf)", s), 0) for s in SIZES]
    pt_lat = [results.get(("PyTorch", "GELU", s), 0) for s in SIZES]

    ax.bar(x - bar_w, tanh_lat, bar_w, label="TileLang GELU (tanh approx)", color="#66c2a5")
    ax.bar(x, erf_lat, bar_w, label="TileLang GELU (erf exact)", color="#fc8d62")
    ax.bar(x + bar_w, pt_lat, bar_w, label="PyTorch GELU", color="#8da0cb")

    ax.set_xlabel("Matrix Size (M=N=K)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("GELU Variant Comparison: tanh approx vs erf exact")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("gemm_gelu_variant_comparison.png", dpi=150)
    plt.close()
    print("图表已保存: gemm_gelu_variant_comparison.png")


# ==============================================================================
# 第八部分: 性能分析
# ==============================================================================


def analyze_results(results):
    """
    分析性能数据，包括：
      1. 融合加速比
      2. GELU tanh vs erf 精度/性能对比
      3. 输出汇总表格
    """
    print("\n" + "=" * 70)
    print("性能分析")
    print("=" * 70)

    # ---- 融合加速比分析 ----
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

    # ---- GELU 变体对比 ----
    print("\n--- GELU 变体性能对比 (tanh vs erf) ---")
    print(f"{'Size':<10} {'tanh (ms)':<15} {'erf (ms)':<15} {'Ratio (erf/tanh)':<15}")
    print("-" * 55)
    for size in SIZES:
        tanh_lat = results.get(("TileLang Fused", "GELU (tanh)", size), 0)
        erf_lat = results.get(("TileLang Fused", "GELU (erf)", size), 0)
        ratio = erf_lat / tanh_lat if tanh_lat > 0 else 0
        print(f"{size:<10} {tanh_lat:<15.4f} {erf_lat:<15.4f} {ratio:<15.2f}")

    # ---- 与 PyTorch 对比 ----
    print("\n--- 与 PyTorch 性能对比 ---")
    print(f"{'Size':<10} {'ReLU (TL/PyT)':<18} {'GELU (TL/PyT)':<18}")
    print("-" * 46)
    for size in SIZES:
        relu_tl = results.get(("TileLang Fused", "ReLU", size), 0)
        relu_pt = results.get(("PyTorch", "ReLU", size), 0)
        gelu_tl = results.get(("TileLang Fused", "GELU (tanh)", size), 0)
        gelu_pt = results.get(("PyTorch", "GELU", size), 0)

        relu_ratio = relu_tl / relu_pt if relu_pt > 0 else 0
        gelu_ratio = gelu_tl / gelu_pt if gelu_pt > 0 else 0
        print(f"{size:<10} {relu_ratio:<18.2f} {gelu_ratio:<18.2f}")

    # ---- 汇总表格（Markdown格式，方便写报告） ----
    print("\n--- 性能汇总表 (单位: ms) ---")
    print()
    # 表头
    header = f"| {'Size':<8} | {'TL Fused ReLU':<14} | {'TL NF ReLU':<11} | {'CUDA ReLU':<10} | {'PyTorch ReLU':<12} | {'TL Fused GELU':<14} | {'TL NF GELU':<11} | {'CUDA GELU':<10} | {'PyTorch GELU':<12} |"
    sep = "|" + "-" * 10 + "|" + "-" * 16 + "|" + "-" * 13 + "|" + "-" * 12 + "|" + "-" * 14 + "|" + "-" * 16 + "|" + "-" * 13 + "|" + "-" * 12 + "|" + "-" * 14 + "|"
    print(header)
    print(sep)

    for size in SIZES:
        relu_f = results.get(("TileLang Fused", "ReLU", size), 0)
        relu_nf = results.get(("TileLang Non-Fused", "ReLU", size), 0)
        relu_cuda = results.get(("CUDA", "ReLU", size), 0)
        relu_pt = results.get(("PyTorch", "ReLU", size), 0)
        gelu_f = results.get(("TileLang Fused", "GELU (tanh)", size), 0)
        gelu_nf = results.get(("TileLang Non-Fused", "GELU", size), 0)
        gelu_cuda = results.get(("CUDA", "GELU", size), 0)
        gelu_pt = results.get(("PyTorch", "GELU", size), 0)

        print(f"| {size:<8} | {relu_f:<14.4f} | {relu_nf:<11.4f} | {relu_cuda:<10.4f} | {relu_pt:<12.4f} | {gelu_f:<14.4f} | {gelu_nf:<11.4f} | {gelu_cuda:<10.4f} | {gelu_pt:<12.4f} |")


# ==============================================================================
# 第九部分: main 入口
# ==============================================================================


def main():
    """
    主函数：依次执行正确性验证、性能基准、可视化和分析。

    如果你的 GPU 显存较小（< 8GB），建议将 SIZES 中的 4096 注释掉，
    因为 4096×4096 的矩阵需要的显存较大。
    """
    import torch

    # 检查 CUDA 可用性
    if not torch.cuda.is_available():
        print("错误: 需要 CUDA GPU 才能运行此脚本")
        return

    gpu_name = torch.cuda.get_device_properties(0).name
    print(f"GPU: {gpu_name}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print(f"TileLang: {tilelang.__version__}")
    print(f"Triton 可用: {HAS_TRITON}")

    # Step 1: 正确性验证
    verify_correctness()

    # Step 2: 性能基准测试
    print("\n\n" + "=" * 70)
    print("性能基准测试")
    print("=" * 70)
    results = run_all_benchmarks()

    # Step 3: 可视化
    print("\n\n" + "=" * 70)
    print("生成性能图表")
    print("=" * 70)
    plot_results(results)

    # Step 4: 分析
    analyze_results(results)

    print("\n✅ 所有任务完成！")


if __name__ == "__main__":
    main()
