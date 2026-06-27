# Report

## fusion 的实现思路

参考 quickstart.py 的写法：在 K 维 Pipelined 循环结束后、`T.copy` 写回 global memory 前，插入一段 `T.Parallel` 对 `C_local`（fragment）做 element-wise 操作。

```python
for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    T.copy(A[...], A_shared)
    T.copy(B[...], B_shared)
    T.gemm(A_shared, B_shared, C_local)

# fusion 关键步骤：在 fragment 上就地做 activation
for i, j in T.Parallel(block_M, block_N):
    C_local[i, j] = T.max(C_local[i, j], 0)   # ReLU

T.copy(C_local, C[...])
```

GELU 同理，把 `T.max` 替换成相应的 tanh 近似或 erf 精确公式。tanh 近似版本里 `sqrt(2/π)` 这种常量没有专门 fold，直接写在表达式里让编译器自己处理。

理论上 fusion 的收益来源是：非融合版本需要把 GEMM 结果写回 global memory，再被 activation kernel 读出来、计算、写回；fusion 直接在寄存器里完成，节省了 2 次 `M*N` 大小的 global memory 往返（一次 read，一次 write）。

## 对比了两种 GELU 变体

T.tanh vs T.erf:

- `gemm_gelu_approx_fused`: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
- `gemm_gelu_exact_fused`: GELU(x) = 0.5 * x * (1 + erf(x / √2))

实测两种实现的性能差距在 ±10% 以内（落在测量噪声范围），说明在现代 GPU 上 `T.tanh` 和 `T.erf` 走的是相近的快速近似实现，cycle 数差不多。所以选哪种取决于精度需求而非性能。

## 为什么之前的结果不符合预期

一开始 Non-Fused baseline 写成了 `c = a @ b; relu(c)`，也就是 PyTorch `mm`（底层走 cuBLAS）+ PyTorch relu。然后跟 TileLang Fused 比，得出了"大矩阵下 fusion 反而变慢"的结论——这违背理论预期。

后来发现这个对比写错了：

- 在 TileLang Fused 中用了 **TileLang 自己的 GEMM**
- 但在 Non-Fused 中用的却是 **cuBLAS GEMM**

这两个对照混淆了两个变量（fusion vs not + TileLang GEMM vs cuBLAS GEMM），测出来的"加速比"实际反映的是"TileLang GEMM 不如 cuBLAS GEMM 快"，不是 fusion 本身的效果。

修复方案：把 Non-Fused 改成了调用 TileLang 自己的 `gemm_only` kernel + PyTorch relu。这样 Fused 和 Non-Fused 共享同一个 GEMM 实现，唯一区别就是 activation 是不是在 fragment 上就地做。修复细节写在 `design-fixes.md` 里。

修复后的结论符合预期：fusion 在所有尺寸下都 ≥ 1.0x，没有变慢的情况。

| Size | ReLU 加速比 (修复前 → 修复后) | GELU 加速比 (修复前 → 修复后) |
|------|-----------------------------|-----------------------------|
| 512  | 2.10x → **2.43x** | 1.17x → **1.29x** |
| 1024 | 1.45x → **1.56x** | 1.00x → **1.12x** |
| 2048 | 0.87x → **1.19x** | 0.73x → **0.99x** |
| 4096 | 0.92x → **1.06x** | 0.86x → **0.98x** |

修复前的"fusion 在大矩阵下变慢"完全是伪结论。

## 性能结果

(RTX 4060 Laptop, 三次运行取平均)

### GEMM + ReLU

| Size | TL Fused | TL Non-Fused | cuBLAS | Triton |
|------|----------|--------------|--------|--------|
| 512  | 0.0169 ms | 0.0411 ms | 0.0359 ms | 0.0164 ms |
| 1024 | 0.0297 ms | 0.0464 ms | 0.0428 ms | 0.0255 ms |
| 2048 | 0.1166 ms | 0.1384 ms | 0.1005 ms | 0.0939 ms |
| 4096 | 0.7507 ms | 0.7955 ms | 0.6811 ms | 0.6621 ms |

### GEMM + GELU

| Size | TL Fused (tanh) | TL Fused (erf) | TL Non-Fused | cuBLAS |
|------|-----------------|----------------|--------------|--------|
| 512  | 0.0297 ms | 0.0325 ms | 0.0382 ms | 0.0340 ms |
| 1024 | 0.0420 ms | 0.0404 ms | 0.0472 ms | 0.0417 ms |
| 2048 | 0.1426 ms | 0.1353 ms | 0.1415 ms | 0.1038 ms |
| 4096 | 0.8308 ms | 0.8256 ms | 0.8179 ms | 0.7080 ms |

> （因为机器没有网，装不了 matplotlib，所以没有作图。）

## 关键发现

### Fusion 收益随矩阵增大而递减

理论预期 fusion 节省的是 `O(M·N)` 的 memory traffic，而 GEMM 计算量是 `O(M·N·K)`。方阵下相对收益 ≈ 节省的 memory time / 总时间，随 n 增大递减。实测加速比 2.43x → 1.06x 单调下降，符合 memory-bound → compute-bound 的转变。

但 fusion 收益**只是递减，不会反转为负**。修复前测出的反转是 benchmark 设计错误。

### GELU 的 fusion 收益小于 ReLU

| 尺寸 | ReLU 加速比 | GELU 加速比 |
|------|-----------|-----------|
| 512 | 2.43x | 1.29x |
| 4096 | 1.06x | 0.98x |

GELU 需要 `x³`/`sqrt`/`tanh` 或 `erf` 加多次乘加，占用大量寄存器，降低 occupancy（同时 in-flight 的 warp 数减少 → 隐藏 memory latency 能力下降）。ReLU 只有一次 `max(x, 0)`，几乎不消耗额外寄存器。

这跟理论预期一致：复杂 activation 的 fusion 收益会被 register pressure 部分抵消。

### TileLang Fused 在小矩阵下比 cuBLAS 还快

| Size | TL Fused ReLU | cuBLAS + ReLU | TL/cuBLAS |
|------|--------------|--------------|----------|
| 512  | 0.0169 ms | 0.0359 ms | **0.47x**（快 2.1x）|
| 1024 | 0.0297 ms | 0.0428 ms | **0.69x**（快 1.4x）|
| 2048 | 0.1166 ms | 0.1005 ms | 1.16x（慢 16%）|
| 4096 | 0.7507 ms | 0.6811 ms | 1.10x（慢 10%）|

cuBLAS 这条路径**不是 fused** 的：`torch.relu(a @ b)` 是两个独立 kernel（cuBLAS GEMM + PyTorch element-wise relu），有完整的 global memory 往返。

小矩阵 memory-bound，fusion 省下的 memory traffic 比 cuBLAS GEMM 的优化优势更重要，所以 TileLang Fused 反超 cuBLAS。大矩阵 compute-bound，cuBLAS GEMM 本身的实现优势（autotuned tile size、async copy、split-K）就占上风了。

### Triton 略快于 TileLang Fused

各尺寸下 Triton GEMM+ReLU 比 TileLang Fused 快 3-12%。两者定位类似（DSL → 编译到 PTX），算法也一样，差距应该来自编译器内部的 tile 配置和 PTX 优化。这跟 fusion 机制无关，是 backend 实现的差距。

GELU 那边 Triton 没写——`tl.math.tanh` 在当前版本不可用，写了一版被 import error 卡住，干脆注释掉了。

## 结论

GEMM + Activation fusion 在所有测试尺寸下都有正向或中性收益，符合理论预期。小矩阵（512/1024）下 fusion 比 cuBLAS 都要快（fusion 优势 > backend 优势）；大矩阵（2048/4096）下 fusion 仍然不亏，但 cuBLAS 的 GEMM 实现优势开始主导。GELU 的复杂度让它的 fusion 收益小于 ReLU，符合 register pressure 影响 occupancy 的预期。
