# 高级编译器课程设计期末大作业方案

## 项目方向

基于 **TileLang** 实现算子编译与性能调优（两人小组，为期一个月）。

---

## 关键词

算子编译，TileLang，GEMM，算子融合，Norm，Attention，自动调优

## 项目背景

大模型的高效运行依赖高性能算子。传统手写 CUDA 开发门槛高、适配新硬件困难。算子编译器通过分离计算描述与调度策略，自动生成优化代码。TileLang 是面向深度学习算子的领域专用语言，构建于 TVM 之上，提供 Pythonic 语法的多级抽象（硬件无关 → 硬件感知分块 → 线程原语），适合理解和实践算子编译核心流程。

## 项目内容

基于 TileLang 复现典型 LLM 算子的编译与优化过程：

1. 使用 Level 2 抽象（分块 + 共享内存）实现 GEMM 算子的编译与正确性验证
2. 实现算子融合（Norm + GEMM 及 GEMM + Activation），与 CUDA / Triton / PyTorch 做多维度性能对比
3. 各自实现一种 Flash Attention 算子（MHA / GQA）
4. 使用 TileLang 内置自动调优工具，对分块参数进行搜索，获得性能对比数据

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

## 阶段二：算子融合与多后端性能对比（6/5 – 6/12）

目标：各自实现一种算子融合，并与 CUDA、Triton、PyTorch、TileLang 非融合版本做性能对比。

### 同学 A：Norm + GEMM 融合

- 融合 RMSNorm（或 LayerNorm）与 GEMM，参考 `examples/norm/rms_norm.py` 和 `examples/gemm/example_gemm.py`
- 融合策略：在 fragment / shared memory 上完成 Norm 计算后直接送入 GEMM，避免将中间结果写回 global memory
- 实验方案：
  1. 实现 RMSNorm + GEMM 融合 kernel
  2. 分别在不同矩阵大小（512/1024/2048/4096）下测试
  3. 性能对比（至少 5 条线）：
     - TileLang 融合版本
     - TileLang 非融合版本（Norm kernel + GEMM kernel 两次调用）
     - CUDA 实现（PyTorch `F.rms_norm` + `torch.mm`）
     - Triton 实现（手写或参考开源实现）
     - PyTorch 原生（`F.rms_norm` + `torch.mm`）
  4. 记录每种配置的延迟，分析融合收益（减少了多少次 global memory 读写）

### 同学 B：GEMM + Activation 融合

- 参考 `examples/quickstart.py`（GEMM + ReLU），在 `T.gemm` 之后、`T.copy` 写回之前，用 `T.Parallel` 循环 + `T.max(x, 0)` 插入激活函数
- 实现 GEMM + ReLU 以及 GEMM + GELU（使用 `T.tanh` 或 `T.erf` + `T.cast`）两种融合
- 理解融合的意义：在 fragment 上就地计算，避免中间结果写回 global memory 再读出
- 实验方案：
  1. 实现 GEMM + ReLU 和 GEMM + GELU 融合 kernel
  2. 分别在不同矩阵大小（512/1024/2048/4096）下测试
  3. 性能对比（至少 5 条线）：
     - TileLang 融合版本
     - TileLang 非融合版本（GEMM kernel + Activation kernel 两次调用）
     - CUDA 实现（`torch.mm` + `F.relu` / `F.gelu`）
     - Triton 实现（手写或参考开源实现）
     - PyTorch 原生（`torch.mm` + `F.relu` / `F.gelu`）
  4. 记录每种配置的延迟，对比 ReLU 和 GELU 融合的收益差异

### 阶段二交付物

- [ ] 同学 A：RMSNorm + GEMM 融合 kernel 可运行，正确性验证通过
- [ ] 同学 B：GEMM + ReLU / GELU 融合 kernel 可运行，正确性验证通过
- [ ] 五维度性能对比表格 + 柱状图：TileLang Fused / TileLang Non-Fused / CUDA / Triton / PyTorch
- [ ] 中期检查：可展示的融合算子 + 性能数据

---

## 阶段三：Flash Attention 算子实现

目标：两位同学各实现一种 Flash Attention 变体，基于 TileLang 从零编写、理解算法原理、对比性能。

### 选题分配

| 同学 | 选题 | Layout | 参考文件 | 说明 |
|------|------|--------|---------|------|
| 同学 A（Norm+GEMM） | **Flash Attention — MHA Forward** | BHSD `[batch, heads, seq, dim]` | `examples/flash_attention/example_mha_fwd_bhsd.py` | 标准多头注意力，每个 Q head 对应一个 KV head |
| 同学 B（GEMM+Activation） | **Flash Attention — GQA Forward** | BSHD `[batch, seq, heads, dim]` | `examples/flash_attention/example_gqa_fwd_bshd.py` | 分组查询注意力（Llama/Mistral 实际使用的模式），KV heads < Q heads |

> **为什么不选其他 Attention？**
> - Online Softmax Attention：现有例子仅做 softmax，做完整 attention 等同于重新推导 FA1，不如直接做 FA1。
> - Flash Attention 2：改进在 warp 调度和寄存器分配，属于编译器/调度器层面，TileLang DSL 层面无法体现区别。
> - Linear / Sparse Attention：数学基础不同，调试困难，且 baseline 难以选择。

### 阶段三A：同学 A — MHA Flash Attention Forward

#### 3A.1 算法原理（理解后再写代码）

Flash Attention 的核心思想：**将 softmax 计算分解为分块 online 更新，避免 O(N²) 的注意力矩阵写回 HBM**。

标准 Attention：
```
S = Q @ K^T          # [seq, seq] ← 这个矩阵是瓶颈
P = softmax(S)
O = P @ V
```

Flash Attention 将 Q 按行分块（`block_M`），K/V 按列分块（`block_N`），在外循环迭代 KV 块时维护三个 running state：

- `acc_o[i]` — 当前行的部分输出累加（fp32）
- `scores_max[i]` — 当前行的 running max（用于数值稳定性）
- `logsum[i]` — 当前行的 running sum（用于最终归一化）

对每个新的 KV 块：
1. 计算局部 `S_block = Q_block @ K_block^T`
2. 更新 `scores_max_new = max(scores_max_old, rowmax(S_block))`
3. 计算 rescale 因子：`scale = exp(scores_max_old - scores_max_new)`
4. 用 `scale` 修正旧的 `acc_o`：`acc_o *= scale`
5. 计算局部 softmax：`P_block = exp(S_block - scores_max_new)`
6. 累加输出：`acc_o += P_block @ V_block`
7. 更新 `logsum_new = logsum_old * scale + rowsum(P_block)`

循环结束后：`Output = acc_o / logsum`

**关键洞察**：整个过程 Q 块只加载一次，K/V 块在循环中依次流经 shared memory，`S` 矩阵（`[block_M, block_N]`）始终在寄存器中——**永远不会写出 `[seq, seq]` 的完整注意力矩阵到 HBM**。

#### 3A.2 TileLang 实现要点

**Layout & Indexing（BHSD）**：
```python
q_shape = [batch, heads, seq_q, dim]   # BHSD
kv_shape = [batch, heads, seq_kv, dim]  # BHSD

# Kernel 3D grid: (seq_q 分块, heads, batch)
with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
    # Q 块索引: bz=batch, by=head, bx=seq_q_block
    T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
    # K 块索引: bz=batch, by=head, k=seq_kv_block
    T.copy(K[bz, by, k * block_N : (k + 1) * block_N, :], K_shared)
```

**内存分配**：
| Buffer | 位置 | Shape | dtype | 用途 |
|--------|------|-------|-------|------|
| `Q_shared` | Shared Mem | `[block_M, dim]` | fp16 | Q 块缓存 |
| `K_shared` | Shared Mem | `[block_N, dim]` | fp16 | K 块缓存（流水线中重用） |
| `V_shared` | Shared Mem | `[block_N, dim]` | fp16 | V 块缓存（流水线中重用） |
| `O_shared` | Shared Mem | `[block_M, dim]` | fp16 | 输出写回缓冲（保证合并写） |
| `acc_s` | Register | `[block_M, block_N]` | fp32 | 局部注意力分数 S |
| `acc_s_cast` | Register | `[block_M, block_N]` | fp16 | S 的 fp16 版本（给 V gemm 用） |
| `acc_o` | Register | `[block_M, dim]` | fp32 | 输出累加器 |
| `scores_max` | Register | `[block_M]` | fp32 | 每行当前最大值 |
| `scores_max_prev` | Register | `[block_M]` | fp32 | 上一轮最大值 |
| `scores_scale` | Register | `[block_M]` | fp32 | rescale 因子 |
| `scores_sum` | Register | `[block_M]` | fp32 | 每行局部 softmax sum |
| `logsum` | Register | `[block_M]` | fp32 | 每行 running log-sum |

**核心循环结构**：
```python
T.copy(Q[...], Q_shared)
T.fill(acc_o, 0)
T.fill(logsum, 0)
T.fill(scores_max, -T.infinity(accum_dtype))

for k in T.Pipelined(num_kv_blocks, num_stages=2):
    T.copy(K[...], K_shared)         # Stage 1: 加载 K

    # QK^T → acc_s，结果在寄存器中
    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

    # Online Softmax rescaling
    T.copy(scores_max, scores_max_prev)
    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
    for i in T.Parallel(block_M):
        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
        scores_scale[i] = T.exp2(scores_max_prev[i] - scores_max[i])
    for i, j in T.Parallel(block_M, dim):
        acc_o[i, j] *= scores_scale[i]
    for i, j in T.Parallel(block_M, block_N):
        acc_s[i, j] = T.exp2(acc_s[i, j] - scores_max[i])

    T.copy(acc_s, acc_s_cast)        # fp32 → fp16 cast
    T.copy(V[...], V_shared)         # Stage 2: 加载 V
    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

    T.reduce_sum(acc_s, scores_sum, dim=1)
    for i in T.Parallel(block_M):
        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

# 最终归一化 + 写回
for i, j in T.Parallel(block_M, dim):
    acc_o[i, j] /= logsum[i]
T.copy(acc_o, O_shared)
T.copy(O_shared, Output[...])
```

**关键 API 说明**：
- `T.gemm(..., transpose_B=True)` — K 需要转置，Q[block_M,dim] × K^T[dim,block_N] → S[block_M,block_N]
- `policy=T.GemmWarpPolicy.FullRow` — 每个 warp 负责一整行，适合 attention 中 block_N 较小的情况
- `num_stages=2` — 双缓冲流水线：加载下一个 K 块的同时计算当前 V 的 gemm
- `scale = (1.0 / dim) ** 0.5 * 1.44269504` — 用 `exp2/log2` 替代 `exp/log`，因子 `1.44269504 = log2(e)` 用于换底；硬件上 `exp2` 比 `exp` 快
- `T.exp2(x)` 返回 fp32，无论输入类型

#### 3A.3 参考 baseline

| Baseline | 实现方式 |
|----------|---------|
| PyTorch native | `F.scaled_dot_product_attention(Q, K, V)` — PyTorch 2.0+ 自动调用 cuDNN Flash Attention |
| PyTorch manual | `einsum + softmax + einsum`（无融合，用于展示融合收益） |
| Triton | 参考 [Triton Fused Attention Tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) |
| flash_attn (可选) | `pip install flash-attn`，调用 `flash_attn_func(Q, K, V)` |

---

### 阶段三B：同学 B — GQA Flash Attention Forward

#### 3B.1 算法原理

Grouped Query Attention (GQA) 与 MHA 的区别：**KV heads 数量少于 Q heads 数量**，多个 Q heads 共享同一组 KV heads。

```
MHA:   Q_heads = 32,  KV_heads = 32   →  每个 Q head 对应一个 KV head
GQA:   Q_heads = 32,  KV_heads = 8    →  每 4 个连续的 Q heads 共享一个 KV head
```

这是 **Llama 2 70B、Llama 3、Mistral、Qwen** 等实际部署的大模型使用的模式。好处：减少 KV cache 大小，几乎不影响模型质量。

在 TileLang 中的核心变化——KV 的索引按 `kv_head = q_head // num_groups` 映射：

```python
num_groups = heads // kv_heads   # 例如 32 // 8 = 4

with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
    kv_head = by // num_groups   # Q head "by" 映射到对应的 KV head
    T.copy(Q[bz, by, seq_start:seq_end, :], Q_shared)
    T.copy(K[bz, kv_head, k_start:k_end, :], K_shared)   # ← 注意 kv_head 索引
    T.copy(V[bz, kv_head, k_start:k_end, :], V_shared)
```

除此之外，online softmax 的 rescaling 逻辑与 MHA 完全一致。

#### 3B.2 TileLang 实现要点

**Layout（BSHD）**：
```python
q_shape = [batch, seq_q, heads, dim]       # BSHD
kv_shape = [batch, seq_kv, kv_heads, dim]  # BSHD, 注意 kv_heads < heads

# Kernel 3D grid: (seq_q 分块, heads, batch)
# by 遍历所有 Q heads，kv_head 通过 by // num_groups 计算
with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
    kv_head = by // num_groups
    T.copy(Q[bz, bx * block_M : (bx+1) * block_M, by, :], Q_shared)
    T.copy(K[bz, k * block_N : (k+1) * block_N, kv_head, :], K_shared)
```

**与 MHA 的差异总结**：

| 方面 | MHA (同学 A) | GQA (同学 B) |
|------|-------------|-------------|
| Layout | BHSD | BSHD |
| KV head 数量 | = Q heads | < Q heads |
| KV 索引 | `by` | `by // num_groups` |
| Q 索引 | `[bz, by, ...]` | `[bz, ..., by]` |
| 实际意义 | 教科书式 | 工业部署实际使用 |
| bonus | — | 可加 causal mask 实现 decoder 模式 |

#### 3B.3 Causal Mask（加分项）

GQA 通常用于自回归解码，加上 causal mask 更有实际意义：

```python
# 在 Pipelined 循环内，gemm 之前：
if is_causal:
    for i, j in T.Parallel(block_M, block_N):
        q_idx = bx * block_M + i
        k_idx = k * block_N + j
        acc_s[i, j] = T.if_then_else(q_idx >= k_idx, 0, -T.infinity(acc_s.dtype))
```

加上 causal mask 后循环上界也可以优化——Q 的第 `bx` 块只需要关注 `k_idx <= q_idx_max` 的 KV 块（后面的块全是 masked）。

---

### 实验方案（两人共用）

#### 测试矩阵

```python
batch_sizes = [1, 2]
seq_lens = [512, 1024, 2048, 4096]
head_dims = [64, 128]
# 同学 A (MHA):  heads = 32
# 同学 B (GQA):  heads = 32, kv_heads = 8
```

#### 性能对比（至少 4 条线）

| # | 名称 | 说明 |
|---|------|------|
| 1 | **TileLang Flash Attn** | 你的实现 |
| 2 | **PyTorch sdpa** | `F.scaled_dot_product_attention(Q, K, V)` — PyTorch 内置 flash attention |
| 3 | **PyTorch manual** | `einsum("bhqd,bhkd->bhqk", Q, K) → softmax → einsum("bhqk,bhkd->bhqd", ...)` — 无融合，用于展示融合收益 |
| 4 | **Triton** | 参考 [Triton Flash Attention Tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)，适配到你的 shape |
| 5 | **flash_attn** (可选) | `pip install flash-attn` → `flash_attn_func(Q, K, V)` — 手写 CUDA 天花板 |

#### 调参建议

```python
# 推荐参数搜索空间（阶段四用 autotune，阶段三先手动试）
block_M = [64, 128]        # Q 行分块大小
block_N = [64, 128]        # KV 列分块大小
num_stages = [1, 2, 3]     # 软件流水线深度
threads = [128, 256]       # 每 block 线程数
```

- 长序列（≥2048）：大的 `block_M`（128）减少循环迭代次数
- 短序列（512）：小的 `block_M`（64）提高 occupancy
- `num_stages=2` 通常是最优平衡点（双缓冲，不明显增加 shared memory 压力）

#### 分析维度

1. **延迟 vs 序列长度**：画出 seq_len 从 512→4096 的延迟曲线，分析复杂度是 O(N²) 还是接近 O(N)（Flash Attention 在带宽-bound 区间接近线性）
2. **融合收益**：对比 PyTorch manual（无融合，O(N²) 中间矩阵写回 HBM）的延迟，量化融合减少的 HBM 读写量
3. **计算强度**（选修）：roofline model 分析——不同 seq_len 下是计算受限还是带宽受限

---

### 阶段三交付物

- [ ] 同学 A：MHA Flash Attention forward pass 可运行，与 `F.scaled_dot_product_attention` 误差 `rtol=1e-2, atol=1e-2`
- [ ] 同学 B：GQA Flash Attention forward pass 可运行，与 `F.scaled_dot_product_attention`（手动处理 head mapping）误差 `rtol=1e-2, atol=1e-2`
- [ ] 不同序列长度（512/1024/2048/4096）的性能对比表格 + 折线图（4-5 条线）
- [ ] 与 PyTorch manual（无融合）的对比，展示 Flash Attention 避免 O(N²) 中间矩阵的收益
- [ ] 导出生成的 CUDA 源码到文件，报告中引用关键代码段
- [ ] 可选：causal mask 版本 + 性能对比

---

## 阶段四：实验总结与报告撰写

> 暂未确定具体分工，以下为初步框架。

### 同学 A：技术章节

整理实验数据，绘制性能对比图表，撰写报告核心技术章节：

- **背景**：算子编译器的作用，TileLang 的定位
- **方法**：TileLang 的分块抽象、软件流水线、tensor core 调度
- **实现细节**：
  - 阶段二的融合算子实现（Norm + GEMM 或 GEMM + Activation），内存优化原理
  - 阶段三的 Flash Attention 实现（MHA / GQA），online softmax 分块 + rescaling 的访存优化原理
- **性能图表**：融合 vs 非融合 vs 多后端对比（柱状图）、Attention 不同序列长度对比（折线图）

推荐使用 matplotlib/seaborn 绘图（已在 `requirements-test.txt` 中）：

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 示例：融合 vs 非融合 vs CUDA vs Triton vs PyTorch 性能对比
backends = ["TileLang\nFused", "TileLang\nNon-Fused", "CUDA", "Triton", "PyTorch"]
latencies = [0.098, 0.145, 0.102, 0.110, 0.105]  # ms
plt.bar(backends, latencies)
plt.ylabel("Latency (ms)")
plt.title("GEMM + ReLU Fusion: Backend Comparison (M=N=K=4096)")
plt.savefig("fusion_backend_comparison.png", dpi=150)
```

### 同学 B：其余章节 + 统稿

- **实验设置**：硬件环境（GPU 型号、CUDA 版本、GPU 显存带宽）、软件版本（PyTorch / Triton / TileLang 版本号）、测试方法
- **结果分析**：
  - 分析融合算子的性能收益（减少了多少次 global memory 读写，带宽节省量）
  - 分析 `num_stages` 对流水线效率的影响
  - 分析 Attention 算子的计算强度和带宽利用率随序列长度的变化
  - 讨论 TileLang 相比手写 CUDA 和 Triton 的优劣势
- **结论**：总结 TileLang 在算子开发中的优缺点
- **挑战与展望**：遇到的问题和解决过程，可能的后续方向

### 最终产出

- 完整的项目代码：
  - 阶段二：`norm_gemm_fusion.py`（同学 A）、`gemm_activation_fusion.py`（同学 B）
  - 阶段二：CUDA / Triton baseline 脚本
  - 阶段三：`mha_flash_attention.py`（同学 A）、`gqa_flash_attention.py`（同学 B）
  - 性能测试脚本 + 绘图脚本
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
| GEMM + GELU 融合 | 参考 `examples/quickstart.py` + `T.tanh` / `T.erf` |
| Norm 实现 | `examples/norm/rms_norm.py` |
| Flash Attention | `examples/flash_attention/` |
| Online Softmax Attention | `examples/online_softmax/online_softmax.py` |
| 装饰器式自动调优 | `examples/convolution/example_convolution_autotune.py` |
| API 式自动调优 | `examples/gemm/example_gemm_autotune.py` |
| 高级调优选项 | `examples/gemm/example_gemm_advanced_autotune.py` |
| GEMM 测试用例 | `testing/python/kernel/test_tilelang_kernel_gemm.py` |
| 自动调优测试用例 | `testing/python/autotune/` |
| Triton GEMM 参考 | <https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html> |
| Triton Flash Attention 参考 | <https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html> |
| TileLang GitHub | <https://github.com/tile-ai/tilelang> |
