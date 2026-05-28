# 高级编译器课程设计期末大作业方案

## 项目方向

基于 **TileLang** 实现算子编译与性能调优（两人小组，为期一个月）。

---

## 关键词

算子编译，TileLang，GEMM，自动调优，算子融合

## 项目背景

大模型的高效运行依赖高性能算子。传统手写 CUDA 开发门槛高、适配新硬件困难。算子编译器通过分离计算描述与调度策略，自动生成优化代码。TileLang 是面向深度学习算子的领域专用语言，构建于 TVM 之上，提供 Pythonic 语法的多级抽象（硬件无关 → 硬件感知分块 → 线程原语），适合理解和实践算子编译核心流程。

## 项目内容

基于 TileLang 复现典型 LLM 算子（GEMM）的编译与优化过程：

1. 使用 Level 2 抽象（分块 + 共享内存）实现 GEMM 算子的编译与正确性验证
2. 实现 GEMM + ReLU 融合算子
3. 使用 TileLang 内置自动调优工具，对分块参数进行搜索，获得性能对比数据

参考对象：TileLang 官方教程及算子示例（https://github.com/tile-ai/tilelang）

---

## 环境搭建

### 1. 克隆仓库并安装依赖

```bash
git clone --recurse-submodules git@github.com:tile-ai/tilelang.git
cd tilelang

# 创建虚拟环境
uv venv --seed .venv        # 或 python3 -m venv .venv
source .venv/bin/activate

# 安装开发依赖
pip install --upgrade pip setuptools wheel
uv pip install --requirements requirements-dev.txt

# 安装 pre-commit（可选，不需要改代码可跳过）
pre-commit install --install-hooks
```

### 2. 编译安装 TileLang（editable 模式）

```bash
pip install --no-build-isolation --verbose --editable .
```

编译时间较长（首次约 10-20 分钟），因为需要从源码构建 TVM + TileLang C++ 扩展。

### 3. 验证安装

```bash
python -c "import tilelang; print(tilelang.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```

预期输出：版本号（如 `0.1.10`）和 `True`。

---

## 阶段一：入门学习（5/30 – 6/4）

目标：掌握 TileLang 基础语法和编译流程，能独立运行和修改示例。

### 学习路线

**第一步：运行 quickstart（30 分钟）**

```bash
python examples/quickstart.py
```

这个脚本包含一个完整的 GEMM + ReLU 融合算子，演示了 TileLang 的全部核心概念。仔细阅读源码 `examples/quickstart.py`，理解每一行的含义：

- `@tilelang.jit` — JIT 编译装饰器，自动将 Python DSL 编译为 CUDA 内核
- `T.const("M, N, K")` — 声明符号常量（从输入 tensor 自动推导）
- `T.Tensor((M, K), dtype)` — 参数类型注解（eager 模式），声明 tensor shape 和 dtype
- `T.empty((M, N), dtype)` — 声明输出 tensor
- `T.Kernel(grid_x, grid_y, threads=128)` — 定义 kernel 启动配置（grid 维度 + 每 block 线程数）
- `T.alloc_shared(shape, dtype)` — 分配共享内存（on-chip shared memory）
- `T.alloc_fragment(shape, dtype)` — 分配寄存器片段（用于 tensor core 计算）
- `T.clear(C_local)` — 清零累加器
- `T.Pipelined(iterations, num_stages=3)` — 软件流水线循环（重叠计算与数据搬运）
- `T.copy(src, dst)` — 数据搬运（global ↔ shared ↔ fragment）
- `T.gemm(A_shared, B_shared, C_local)` — 调用 tensor core 执行矩阵乘法
- `T.Parallel(rows, cols)` — 并行迭代（每个线程处理不同元素）
- `kernel.get_kernel_source()` — 获取生成的 CUDA 源码
- `kernel.get_profiler()` — 获取性能分析器

**第二步：理解 elementwise 示例（30 分钟）**

```bash
python examples/elementwise/example_elementwise_add.py
```

`examples/elementwise/example_elementwise_add.py` 是最简内核，仅做逐元素加法。对照 quickstart 理解简化后的结构：没有 `T.gemm` 和 `T.Pipelined`，只有最基本的 `T.Parallel` + 手动计算。

**第三步：阅读 GEMM 示例及其 README（1 小时）**

```bash
python examples/gemm/example_gemm.py
```

- `examples/gemm/example_gemm.py` — 标准分块 GEMM（不含融合）
- `examples/gemm/README.md` — GEMM 的详细文档，包含高级特性说明

**第四步：浏览更多示例了解算子模式（1 小时）**

推荐按顺序浏览以下文件，了解不同算子的写法模式：

| 文件 | 学什么 |
|------|--------|
| `examples/online_softmax/online_softmax.py` | reduction 操作模式 |
| `examples/norm/rms_norm.py` | RMSNorm + split-K 变体 |
| `examples/gemv/example_gemv.py` | 矩阵-向量乘法模式 |

### 阶段一交付物

- [ ] `quickstart.py` 成功运行，输出 `Kernel output matches PyTorch reference.`
- [ ] `example_gemm.py` 成功运行
- [ ] 两人都能口头解释：T.Kernel、T.copy、T.gemm、T.Pipelined、T.alloc_shared、T.alloc_fragment 的作用

---

## 阶段二：GEMM 算子复现（6/5 – 6/12）

目标：自主实现 GEMM 算子，完成正确性验证和基准性能测试。

### 同学 A：GEMM 实现

- 参考 `examples/gemm/example_gemm.py`，自己重写一个 GEMM kernel（不要直接复制），加深对 Tiling + Pipelining + Tensor Core 调度流程的理解
- 核心模式：分配 shared memory → 清零 fragment → Pipelined 循环（copy A/B to shared → gemm）→ copy 结果回 global
- 实验方案：
  1. 固定 block 参数，测试不同矩阵大小（512/1024/2048/4096）
  2. 固定矩阵大小，测试不同 block 参数（block_M/N: 64/128/256; block_K: 16/32/64）
  3. 记录每种配置的延迟，分析分块参数对性能的影响

### 同学 B：正确性测试与基准性能

- 使用 `kernel.get_profiler()` 获取 Profiler 对象，调用 `profiler.assert_allclose(ref_program=...)` 以 PyTorch `a @ b` 为参考验证正确性
- 使用 `profiler.do_bench(backend="cupti")` 或 `tilelang.profiler.do_bench()` 测延迟，backend 可选 `"cupti"`（更精确）或 `"event"`
- 用 `kernel.get_kernel_source()` 查看生成的 CUDA 源码，对照理解编译结果
- 输出 TileLang vs PyTorch 延迟对比及 speedup

### 阶段二交付物

- [ ] 自主实现的 GEMM kernel 可运行，与 PyTorch 误差在 `rtol=1e-2` 以内
- [ ] 不同 M/N/K 和不同 block 大小的性能对比表格
- [ ] 中期检查：可展示的 GEMM 复现结果 + 性能数据

---

## 阶段三：算子融合与自动调优（6/16 – 6/22）

### 同学 A：GEMM + ReLU 融合

- 参考 `examples/quickstart.py`（已包含 GEMM + ReLU），在阶段二的 GEMM 基础上，在 `T.gemm` 之后、`T.copy` 写回之前，用 `T.Parallel` 循环 + `T.max(x, 0)` 插入 ReLU 激活
- 理解融合的意义：在 fragment 上就地计算，避免中间结果写回 global memory 再读出
- 选做：尝试 GEMM + GELU 融合（使用 `T.tanh` 和 `T.cast`）

### 同学 B：自动调优

- TileLang 提供两种调优方式：
  - **装饰器模式**：在 `@tilelang.jit` 上添加 `@tilelang.autotune(configs=...)`，`compile()` 时自动搜索最佳配置。参考 `examples/convolution/example_convolution_autotune.py`
  - **AutoTuner API**：`AutoTuner.from_kernel()` → `set_compile_args()` → `set_profile_args()` → `run()`，更灵活。参考 `examples/gemm/example_gemm_autotune.py` 和 `examples/gemm/example_gemm_advanced_autotune.py`
- 建议使用 AutoTuner API，搜索 7-15 个配置（block_M/N/K 和 num_stages 的组合），对比最佳结果与手动选参的性能差异
- 可配置环境变量 `TILELANG_AUTO_TUNING_DISABLE_CACHE=1` 禁用缓存确保每次重新搜索，`TILELANG_AUTO_TUNING_CPU_COUNTS=4` 控制并行线程数
- 参考测试用例 `testing/python/autotune/` 了解 API 的正确用法

### 阶段三交付物

- [ ] GEMM + ReLU 融合 kernel 可运行，正确性验证通过
- [ ] 自动调优脚本可运行，输出最佳配置和性能数据
- [ ] 性能对比：手动参数 vs 自动调优最优参数 vs PyTorch，至少 3 种矩阵大小

---

## 阶段四：实验总结与报告撰写（6/23 – 6/28）

### 同学 A：技术章节

整理实验数据，绘制性能对比图表，撰写报告核心技术章节：

- **背景**：算子编译器的作用，TileLang 的定位
- **方法**：TileLang 的分块抽象、软件流水线、tensor core 调度
- **实现细节**：GEMM 的 tiling 策略（global → shared → fragment），融合算子的内存优化原理（避免中间结果写回 global memory）
- **性能图表**：不同分块参数的延迟对比（柱状图）、TileLang vs PyTorch 性能对比（柱状图）

推荐使用 matplotlib/seaborn 绘图（已在 `requirements-test.txt` 中）：

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 示例：不同 block_M 的延迟对比
configs = ["64", "128", "256"]
latencies = [0.123, 0.098, 0.115]  # ms
plt.bar(configs, latencies)
plt.xlabel("block_M")
plt.ylabel("Latency (ms)")
plt.savefig("block_M_comparison.png", dpi=150)
```

### 同学 B：其余章节 + 统稿

- **实验设置**：硬件环境（GPU 型号、CUDA 版本）、软件版本、测试方法
- **结果分析**：分析分块大小如何影响 occupancy 和 memory bandwidth；分析融合算子的性能收益（减少了多少次 global memory 读写）；分析 `num_stages` 对流水线效率的影响
- **结论**：总结 TileLang 在算子开发中的优缺点
- **挑战与展望**：遇到的问题和解决过程，可能的后续方向（如扩展到 Flash Attention）

### 最终产出

- 完整的项目代码（`our_gemm.py`, `matmul_relu.py`, `autotune_experiment.py`）
- 项目报告（含性能图表）
- 可选：PPT / Poster

---

## 常见问题与调试技巧

### 编译错误

```bash
# 查看详细编译日志
TILELANG_VERBOSE=1 python our_gemm.py

# 导出生成的 CUDA 源码到文件
python -c "
from our_gemm import matmul
k = matmul.compile(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32)
print(k.get_kernel_source())
" > kernel.cu
```

### 查看 IR 中间表示

```python
from tilelang.transform import PassConfigKey

kernel = matmul.compile(
    M=1024, N=1024, K=1024,
    block_M=128, block_N=128, block_K=32,
    pass_configs={
        PassConfigKey.TL_ENABLE_DUMP_IR: True,
        PassConfigKey.TL_DUMP_IR_PATH: "./dump_ir/",
    }
)
```

### 运行测试验证

```bash
# 运行 GEMM 相关测试
pytest testing/python/kernel/test_tilelang_kernel_gemm.py -v

# 运行单个测试
pytest testing/python/kernel/test_tilelang_kernel_gemm.py -k "test_gemm_f16" -v

# 运行自动调优测试（了解正确的 API 用法）
pytest testing/python/autotune/ -v
```

### 检查 GPU 信息

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_properties(0))"
```

### 参考示例索引

| 需求 | 参考文件 |
|------|----------|
| 最简内核结构 | `examples/elementwise/example_elementwise_add.py` |
| 标准 GEMM | `examples/gemm/example_gemm.py` |
| GEMM + ReLU 融合 | `examples/quickstart.py` |
| 装饰器式自动调优 | `examples/convolution/example_convolution_autotune.py` |
| API 式自动调优 | `examples/gemm/example_gemm_autotune.py` |
| 高级调优选项 | `examples/gemm/example_gemm_advanced_autotune.py` |
| GEMM 测试用例 | `testing/python/kernel/test_tilelang_kernel_gemm.py` |
| 在线 softmax | `examples/online_softmax/online_softmax.py` |
| RMSNorm | `examples/norm/rms_norm.py` |
