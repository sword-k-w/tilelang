# 中期检查报告：基于 TileLang 的算子编译与融合

> **项目方向：** 基于 TileLang 实现算子编译与性能调优
> **当前阶段：** 阶段二 — 算子融合（RMSNorm + GEMM）
> **日期：** 2026-06-11

---

## 1. 环境信息

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| SM 数量 | 24 |
| Shared Memory / Block | 48 KB |
| Shared Memory / SM | 100 KB |
| Max Threads / SM | 1536 |
| CUDA | 8.9 |
| PyTorch | 2.12.0+cu130 |
| TileLang | 0.1.10+cuda |

---

## 2. 已完成工作

### 2.1 阶段一：基础学习（5/30 – 6/4）

通读了 TileLang 全部核心示例，掌握了 DSL 编程模型：

| 示例 | 学到的知识 |
|------|-----------|
| `examples/quickstart.py` | GEMM + ReLU 融合、T.Kernel、T.Pipelined、T.gemm |
| `examples/elementwise/example_elementwise_add.py` | T.Parallel 逐元素操作 |
| `examples/gemm/example_gemm.py` | 标准分块 GEMM、T.Pipelined 流水线 |
| `examples/online_softmax/online_softmax.py` | T.reduce_max / T.reduce_sum、两 pass 算法 |
| `examples/norm/rms_norm.py` | RMSNorm、split-K 模式、T.rsqrt |
| `examples/gemv/example_gemv.py` | T.get_thread_binding、T.atomic_add、T.vectorized |
| `examples/gemm/README.md` | Lazy 模式、Swizzle、Fine-grained MMA |

关键学习成果：
- 理解 `@tilelang.jit` 的 Eager/Lazy 两种模式和两阶段编译流程
- 掌握 T.Kernel / T.copy / T.gemm / T.Pipelined / T.Parallel / T.reduce_* 等核心原语
- 理解 GPU 的三级内存层次（Global → Shared → Fragment）及对应的 DSL 抽象

### 2.2 阶段二：算子融合 — RMSNorm + GEMM（6/5 – 6/12）

实现了 **RMSNorm + GEMM 融合**，代码位于 `kernel/norm_gemm/norm_gemm_fusion.py`。

#### 融合算法

采用**两 pass 策略**：

```
Pass 1 (T.Serial):    遍历 K 维度分块，在 fragment 上累加 x²
                      → T.reduce_sum 归约 → T.rsqrt 得到 per-row scale

Pass 2 (T.Pipelined): 标准分块 GEMM，利用 scale 对输出做 post-scale
                      (scale[i] × (X @ W) = (scale[i] × X) @ W)
```

#### 关键技术决策

| 决策 | 原因 |
|------|------|
| Post-scale（非 pre-scale） | 避免在 Pipelined 循环中修改 shared memory，绕过 PipelinePlanning 的 stage 写入冲突 |
| `local` 复用（同一 fragment 服务于 Pass 1 和 Pass 2） | 减少 fragment 数量，通过 LayoutInference 检查 |
| `block_K == block_N` | 复用 `local` 的约束：Pass 1 需要 `(block_M, block_K)`，Pass 2 需要 `(block_M, block_N)` |
| `block_M=128, block_N=64, block_K=64` | 在寄存器压力、Shared Memory 占用（48KB 上限）和 GEMM 效率之间取最优 |

#### 面临的技术挑战

1. **LayoutInference 限制：** 两个 2D fragment（`A_local` + `C_local`）会导致 `no available layout found`。解决方案是复用同一个 fragment。
2. **PipelinePlanning 冲突：** Pipelined 循环中 Stage 0（copy）和 Stage 1（compute）不能同时写入同一个 shared memory buffer。解决方案是将 Norm 操作放在 Pipelined 循环外（post-scale）。
3. **atomic_add 性能：** 尝试用 shared memory + atomic 代替 2D fragment 做 x² 累加，但大量原子操作导致 102ms 延迟（vs 8ms）。

#### 性能结果

| 矩阵大小 | TileLang Fused | PyTorch Unfused | Speedup |
|---------|---------------|----------------|---------|
| 512² | **0.0435 ms** | 0.0573 ms | **1.32×** |
| 4096² | **8.24 ms** | 6.85 ms | **0.83×** |

- **小矩阵（512²）：** 融合有效，省去的 Global Memory 读写（1 写 + 1 读）占主导 → 1.32× 加速
- **大矩阵（4096²）：** 融合收益（~1.6ms）被 GEMM 效率差距（~2ms）抵消。根因是 `block_K=64` 导致 Shared Memory 占用 48KB（达到 RTX 4060 Laptop 上限），Occupancy 降至 2 blocks/SM，而 cuBLAS 可达 3 blocks/SM

#### 探索过的替代方案

| 方案 | 结果 |
|------|------|
| Pre-scale（在 Pipelined 内修改 shared memory） | PipelinePlanning 冲突 |
| Atomic 累加 x²（单 pass 融合） | 102 ms，原子操作开销过大 |
| 解耦 `A_local` / `C_local` 两 fragment | LayoutInference 失败 |
| `num_stages=3` 增加 Pipelined 深度 | Shared Memory 溢出 |

---

## 3. 下一步计划

### 阶段二收尾（6/12 前）
- [ ] 实现 Triton 版 RMSNorm + GEMM baseline
- [ ] 对比：TileLang Fused / TileLang Non-Fused / Triton / PyTorch / CUDA
- [ ] 绘制五维度性能对比图表

### 阶段三（6/16 – 6/22）
- [ ] 选择一种 Attention 算子实现（Flash Attention 或 Online Softmax Attention）
- [ ] 与 PyTorch / Triton 做性能对比

---

## 4. 项目结构

```
kernel/
├── README.md                         # 本报告
└── norm_gemm/
    └── norm_gemm_fusion.py           # RMSNorm + GEMM 融合实现
```

---

## 5. 使用方法

```bash
# 小矩阵测试
python kernel/norm_gemm/norm_gemm_fusion.py --M 512 --N 512 --K 512

# 大矩阵测试
python kernel/norm_gemm/norm_gemm_fusion.py --M 4096 --N 4096 --K 4096

# 自定义块大小
python kernel/norm_gemm/norm_gemm_fusion.py --M 2048 --N 2048 --K 2048 \
    --block_M 128 --block_N 64 --block_K 64
```
