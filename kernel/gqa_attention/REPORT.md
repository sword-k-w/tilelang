# GQA Flash Attention — Implementation & Performance Report

## TL;DR

This report documents two contributions to the GQA forward kernel under
`kernel/gqa_attention/gqa_flash_attention.py`:

1. **Autotuning + a fair benchmark harness.** The TileLang-official GQA example
   ships with `@autotune`, but is silently bypassed when tile parameters are
   passed at call sites. Calling the official with explicit tile kwargs (the
   pattern most users default to) disables its search and produces misleading
   "I beat the official" numbers. We fix this and verify, via byte-level CUDA
   diff, that at the same tile config the two kernels are equivalent.
2. **A group-aware GQA kernel** that loads K/V once per KV head and reuses it
   across all `num_groups` Q heads sharing that KV head, instead of reloading
   K/V once per Q head. This is the natural GQA optimization but it is not
   present in any of the existing TileLang flash-attention examples we
   surveyed.

The headline numbers (A800-80GB, `heads=32, kv_heads=8, dim=64, is_causal=False`):

**At default config** (`block_M=64, block_N=64, num_stages=2, threads=128`,
the official's defaults — the regime most users hit when they copy the
example without tuning):

| seq_len | Standard | Group-aware | Official | PyTorch sdpa | GA vs Std | GA vs Off |
|---:|---:|---:|---:|---:|---:|---:|
| 512  | 0.0310 ms | 0.0314 ms | 0.0312 ms | 0.0493 ms | 0.99x | 0.99x |
| 1024 | 0.0772 ms | 0.0744 ms | 0.0775 ms | 0.1004 ms | 1.04x | 1.04x |
| 2048 | 0.2617 ms | 0.2228 ms | 0.2591 ms | 0.2709 ms | **1.17x** | **1.16x** |
| 4096 | 0.9933 ms | **0.8676 ms** | 0.9832 ms | 0.9581 ms | **1.14x** | **1.13x** |

**Both kernels autotuned per shape** (the apples-to-apples comparison —
both search their own config space, pick the best tile, and bench):

| seq_len | Standard | Group-aware | Official | sdpa | GA vs Std | GA vs Off |
|---:|---:|---:|---:|---:|---:|---:|
| 512  | 0.0290 ms | 0.0293 ms | 0.0290 ms | 0.0496 ms | 0.99x | 0.99x |
| 1024 | 0.0708 ms | 0.0711 ms | 0.0714 ms | 0.1010 ms | 1.00x | 1.00x |
| 2048 | 0.2241 ms | 0.2249 ms | 0.2206 ms | 0.2701 ms | 1.00x | 0.98x |
| 4096 | 0.8984 ms | **0.8377 ms** | 0.8634 ms | 0.9597 ms | **1.07x** | **1.03x** |

The two regimes give different headlines. Autotuning the standard kernel
recovers most of the apparent GA win by picking larger Q tiles
(`block_M=128`), which doubles arithmetic intensity per gemm — but it
cannot change the grid topology, so a residual 3–7% advantage remains at
`seq=4096`. At `seq=4096` the autotuned group-aware kernel also beats
PyTorch SDPA (which calls cuDNN's hand-tuned Flash Attention) by ~1.15x.

The first 1.10x "win" we initially measured was actually an artifact of
*disabling* the official's autotune (see section 3); the real story is
that autotuning is a powerful equalizer and the algorithmic win that
survives full tuning is smaller but still non-zero.

---

## 1. Background

### 1.1 Grouped-Query Attention (GQA)

GQA reduces the size of the KV cache (and the bandwidth needed to read it)
by letting multiple Q heads share a single KV head. With `heads` Q heads and
`kv_heads` KV heads, define

```
num_groups = heads // kv_heads
```

Q head `q ∈ [0, heads)` maps to KV head `q // num_groups`. The standard
configuration in Llama-2-70B is `heads=64, kv_heads=8` (`num_groups=8`);
Llama-3 70B uses `heads=64, kv_heads=8` as well; Mistral-7B uses `heads=32,
kv_heads=8`. We use the Mistral shape `(heads=32, kv_heads=8)` throughout
this report.

### 1.2 Flash Attention 1 (recap)

Flash Attention 1 (Dao et al., NeurIPS 2022) computes
`O = softmax(Q K^T / sqrt(d)) V` without materializing the full
`[seq, seq]` attention matrix `S`. It blocks Q by rows (size `block_M`)
and K/V by columns (size `block_N`), then for each Q block iterates over
K/V blocks while maintaining three running statistics per Q row:

- `m` — running max of seen scores (numerical stability)
- `ℓ` — running sum of `exp(scores - m)` (normalization denominator)
- `O` — running unnormalized output accumulator

When a new KV block `k` is processed:

1. `S_k = Q · K_k^T`
2. `m_new = max(m_old, rowmax(S_k))`
3. `α = exp(m_old - m_new)`
4. `O ← α · O + P · V_k` where `P = exp(S_k - m_new)`
5. `ℓ ← α · ℓ + rowsum(P)`

After the last block, the answer is `O / ℓ`. The complete attention matrix
`S` is never written to HBM — only block-sized fragments transit through
shared memory and registers.

### 1.3 Tensor layouts

We use BSHD layout for Q, K, V (`[batch, seq, heads, dim]`). The official
example shipped with TileLang uses the same. This matches the storage layout
emitted by typical inference engines (vLLM, TGI, SGL) for the KV cache.

---

## 2. The baseline (standard) kernel

The baseline kernel in `_build_gqa_prim_func` (lines 92–245 of the kernel
file) is a faithful FA1 implementation. The grid is

```
(seq_q // block_M, heads, batch)
```

with one block per `(Q row block, Q head, batch)` triple, and each block
loads K/V for its kv_head via `kv_head = by // num_groups`. The body
follows the FA1 algorithm exactly. The structure is essentially identical
to the TileLang official example (`examples/flash_attention/example_gqa_fwd_bshd.py`).

We refactored the prim_func body into a free function `_build_gqa_prim_func`
so that the direct-compile entry point (`gqa_flash_attn`) and the autotuned
entry point (`gqa_flash_attn_tuned`) share a single source of truth — see
section 3 for why the autotuned variant exists.

### 2.1 Correctness

The kernel matches PyTorch reference within `rtol=1e-2, atol=1e-2` at all
seq lengths we tested (512 / 1024 / 2048 / 4096). The reference
implementation `ref_gqa` expands KV via `repeat_interleave` then runs
`einsum + softmax + einsum`.

---

## 3. Lesson learned: the autotune fairness trap

This is a methodological note, but it was the most instructive part of the
whole project and is worth recording in detail.

### 3.1 What we initially saw

Our first benchmark, comparing the standard kernel against the official with
`block_M=block_N=64, num_stages=2, threads=128`, showed our kernel at 1.08×
the throughput of the official. We had not changed the algorithm in any
non-trivial way — only renamed a few intermediate variables and slightly
reordered the online-softmax steps — so this was already suspicious.

### 3.2 The hunt for the cause

Two hypotheses were chased:

**(a) The variable reordering matters.** The official does
`acc_o *= scores_scale` after the exp/cast on `acc_s`; we (initially) did
`O_local *= rescale` before the exp on `S_local`. The intuition was that
the FFMA work on `O_local` and the SFU `exp2` work on `S_local` are
independent, so adjacent placement lets the GPU's separate FFMA and SFU
pipelines run in parallel and hides the slow exp latency. This is real in
principle — but when we reverted to the official's exact ordering, our
kernel was still 1.08× faster. So the ordering was not the cause.

**(b) Compiler non-determinism / measurement noise.** We added a CUDA-source
diff to the harness (see `main()` lines ~730–766). When the two kernels are
compiled at the same `(block_M, block_N, num_stages, threads)`, the
generated CUDA differs only in identifier renames (`acc_s` ↔ `S_local`,
`acc_o` ↔ `O_local`, etc.). The PTX structure, register allocation, MMA
schedule, and SMEM layout are identical. So compiler non-determinism was
not the cause either.

The actual cause turned out to be more embarrassing.

### 3.3 The bypass

The official kernel is decorated `@autotune(...) @tilelang.jit(...)`. When
called with **all** tunable parameters supplied as kwargs:

```python
official_gqa_flashattn(
    batch, heads, seq_len, dim, is_causal,
    groups=num_groups,
    block_M=64, block_N=64, num_stages=2, threads=128,   # ← all tunables
)
```

TileLang's autotuner prints

```
WARNING: Tunable parameters ['block_M', 'block_N', 'num_stages', 'threads']
         already provided during auto-tuning. Skipping compilation and using
         direct JIT
```

…and **does not search**. The user gets back a kernel compiled with
exactly those parameters. The `@autotune` decorator becomes a no-op.

Meanwhile our autotuned variant (`gqa_flash_attn_tuned`) was searching its
space and finding `(block_M=64, block_N=128, num_stages=2, threads=128)`
which is ~10% faster than `(64, 64, 2, 128)` for `dim=64`. So our "1.10×
win" was actually:

```
my-tuned (block_N=128)   vs   official-with-tile-kwargs (block_N=64, autotune skipped)
                              ────────────────────────
                              ≠ "the autotuned official"
```

It was apples to oranges, and we were the orange.

### 3.4 The fix

In `main()` and `benchmark_sweep()`, when `--tune` is on, omit the tile
kwargs when calling the official:

```python
kernel_official = official_gqa_flashattn(
    batch, heads, seq_len, dim, is_causal,
    groups=num_groups,
    # no block_M / block_N / num_stages / threads → official's own autotune fires
)
```

When `--tune` is off, both kernels are called with the same (CLI-provided)
tile kwargs and you're comparing apples to apples at a fixed config.

After this fix, with both kernels autotuned, the gap collapses:

```
My GQA Flash Attn (tuned)   0.0293 ms   73.34 TFlops
TileLang Official (tuned)   0.0289 ms   74.30 TFlops
Ratio:                       0.99x
```

i.e., the two kernels are equivalent — exactly as the CUDA diff predicted.

### 3.5 Takeaway

> **An autotuner that silently no-ops when overridden is a worse footgun than
> one that doesn't exist at all,** because it provides false confidence in
> baselines. Anyone benchmarking against `example_gqa_fwd_bshd.py` while
> passing tile kwargs (which is the path of least friction for users) is
> implicitly comparing against an *un-tuned* baseline. We strongly recommend
> that downstream users either (i) call the official without tile kwargs to
> let its autotune run, or (ii) construct an honest baseline that matches
> their search space.

With this fairness fix in place, beating the official actually means
something. That motivates section 4.

---

## 4. The group-aware kernel

### 4.1 The opportunity

In the standard kernel, the grid Y dim is `heads` — one block per Q head.
Every block loads its own copy of K and V (indexed by `kv_head = by // num_groups`).
For our `heads=32, kv_heads=8` setup, this means each KV head is loaded
from HBM `num_groups = 4` times per (Q row block, batch), once for each Q
head in its group.

```
              KV head 0        KV head 1   ...   KV head 7
   block Q0   ─────────┐       ─────────┐       ─────────┐
   block Q1   ─────────┤       ─────────┤       ─────────┤  → 4 × duplicate loads
   block Q2   ─────────┤       ─────────┤       ─────────┤    per KV head, per row block
   block Q3   ─────────┘       ─────────┘       ─────────┘
```

For long sequences, the KV transfer dominates the runtime (the kernel is
bandwidth-bound). The 4× redundant load is therefore a 4× headwind.

This is the *exact* reason GQA exists — Llama et al. introduced GQA to
**shrink the KV cache** so memory bandwidth is no longer a per-Q-head cost.
A kernel that nonetheless loads KV per-Q-head is leaving the central GQA
benefit on the table.

### 4.2 Survey: this optimization is not in the TileLang examples

We checked every flash-attention example shipped with TileLang:

```
examples/flash_attention/example_gqa_fwd_bshd.py
examples/flash_attention/example_gqa_fwd_varlen.py
examples/flash_attention/example_gqa_bwd.py
examples/flash_attention/example_gqa_bwd_tma_reduce.py
examples/flash_attention/example_gqa_bwd_tma_reduce_varlen.py
examples/flash_attention_sm100/gqa_fwd_bshd.py
examples/attention_sink/example_gqa_sink_bwd_bhsd.py
```

All of them index KV via `by // groups` from a grid where Y spans **all**
Q heads. None of them shares a KV load across the Q heads of a group.
Implementing this is the headline contribution of this project.

### 4.3 Algorithm

The grid becomes

```
(seq_q // block_M, kv_heads, batch)
```

— note `kv_heads`, not `heads`. Each block processes the union of
`num_groups` Q heads for one kv_head. The Q tile widens from
`[block_M, dim]` to `[num_groups * block_M, dim]`, packed group-major:

```
Q_shared row index    Q head           Q row in seq
─────────────────────────────────────────────────────
0 .. block_M-1        kv_head*ng + 0   bx*block_M + s, s = i
block_M .. 2bM-1      kv_head*ng + 1   bx*block_M + s, s = i - block_M
2bM .. 3bM-1          kv_head*ng + 2   ...
3bM .. 4bM-1          kv_head*ng + 3   ...
```

All FA fragments (`S_local`, `O_local`, `S_max`, `rescale`, `total_sum`,
…) widen along their row dim to `M_tile = num_groups * block_M`. The
softmax bookkeeping is still per-row, and rows belonging to different
groups are *independent* — there is no cross-row math in the FA inner
loop except `reduce_max` and `reduce_sum`, which operate per row.

K/V tiles stay `[block_N, dim]`. Inside the KV loop:

```python
T.copy(K[bz, k*block_N : (k+1)*block_N, kv_head, :], K_shared)   # ONCE per kv_head
# ... mask, gemm(Q_shared, K_shared, S_local), online softmax ...
T.copy(V[bz, k*block_N : (k+1)*block_N, kv_head, :], V_shared)   # ONCE per kv_head
T.gemm(S_local_fp16, V_shared, O_local)
```

Compared to the standard kernel: same compute (we still produce the same
output rows; we just did it from a different grid block), but the K and V
loads happen `num_groups` times less often *across the whole grid*. Since
there are `num_groups`-fold fewer blocks (Y dim shrank from `heads` to
`kv_heads`), and each surviving block does `num_groups`× the work but
issues exactly the same K/V loads it would have done as a single-head
block, the total HBM traffic for K and V drops by `num_groups`.

### 4.4 The causal mask is invariant across groups

A subtle point: with the group-major row packing, every row of the
super-tile that has the same `s = i % block_M` corresponds to the *same*
Q sequence position (just different Q heads). Therefore the causal
predicate

```
q_idx >= k_idx ?
```

is the same for all `num_groups` rows at the same `s`. The mask code
needs no change:

```python
for i, j in T.Parallel(M_tile, block_N):
    s_in_block = i % block_M
    q_idx = bx * block_M + s_in_block
    k_idx = k * block_N + j
    S_local[i, j] = T.if_then_else(q_idx >= k_idx, 0, -inf)
```

…and the `loop_range` clamp `T.ceildiv((bx+1)*block_M, block_N)` also
remains correct because it depends only on Q row position, not Q head.

### 4.5 Trade-off: register pressure

The price for KV reuse is bigger per-block register fragments. For
`num_groups=4, block_M=64, dim=64, block_N=64`:

| Fragment | Standard size | Group-aware size | Ratio |
|---|---:|---:|---:|
| `S_local` (fp32) | `64×64 = 4K elem` | `256×64 = 16K elem` | 4× |
| `O_local` (fp32) | `64×64 = 4K elem` | `256×64 = 16K elem` | 4× |
| `S_max`, `rescale`, `total_sum`, … (fp32) | 64 each | 256 each | 4× |

The total register-resident state grows by `num_groups`. Modern Ampere/
Hopper SMs have 64K 32-bit registers per SM and the per-thread limit is
255 (with 256-thread blocks); the compiler may need to lower per-thread
allocation, which reduces occupancy.

Mitigation: the group-aware kernel uses *smaller* `block_M` defaults to
keep `M_tile = num_groups * block_M` reasonable. We default to `block_M=32`
in the direct-compile path (vs `block_M=64` for the standard kernel). The
autotuned variant `gqa_flash_attn_group_aware_tuned` searches
`block_M ∈ {16, 32, 64}` (vs `{64, 128}` for the standard kernel).

### 4.6 Q load and output write are explicit `T.Parallel`

The Q source slice has shape `[block_M, num_groups, dim]` (a 3D BSHD
sub-tile) and the Q destination is `[M_tile, dim]` (2D). We do the gather
explicitly:

```python
for s, g, d in T.Parallel(block_M, num_groups, dim):
    Q_shared[g * block_M + s, d] = Q[bz, bx * block_M + s,
                                       kv_head * num_groups + g, d]
```

The output write is the mirror image, routed through `O_shared` for
coalesced writes back to HBM.

---

## 5. Results

### 5.1 Hardware and software

- **GPU:** NVIDIA A800 80GB PCIe (Ampere, SM_80, 108 SMs)
- **CUDA:** 12.4
- **PyTorch:** 2.6.0+cu124
- **TileLang:** 0.1.11 (commit `cf610fce`, after the autotune fairness fix)

### 5.2 Benchmark protocol

All numbers come from `tilelang.profiler.do_bench(fn, warmup=500, rep=100)`.
We bench in this order per shape: standard kernel, group-aware kernel,
official kernel, PyTorch SDPA, PyTorch manual. Each call is GPU-event timed.
Correctness is checked separately against PyTorch reference at every shape;
we report no number whose underlying kernel failed correctness.

### 5.3 Headline result: default-config sweep (no autotuning)

`batch=1, heads=32, kv_heads=8, dim=64, is_causal=False`, both ours and the
official at the same default `(block_M=64, block_N=64, num_stages=2,
threads=128)`. Group-aware uses its own default `(block_M=32, block_N=64,
num_stages=2, threads=128)` since the standard's `(64,64,…)` would produce
`M_tile=256` which is too large at `block_N=64` for our SMEM budget at
the default thread count.

| seq_len | Standard (ms) | Group-aware (ms) | Official (ms) | PyTorch sdpa (ms) | PyTorch manual (ms) | GA vs Std | GA vs Off |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512  | 0.0310 | 0.0314 | 0.0312 | 0.0493 | 0.1012 | 0.99x | 0.99x |
| 1024 | 0.0772 | 0.0744 | 0.0775 | 0.1004 | 0.3659 | 1.04x | 1.04x |
| 2048 | 0.2617 | 0.2228 | 0.2591 | 0.2709 | 1.8469 | **1.17x** | **1.16x** |
| 4096 | 0.9933 | 0.8676 | 0.9832 | 0.9581 | 5.9657 | **1.14x** | **1.13x** |

Notes:
- At `seq=512` the kernel runtime is so short (~30 μs) that launch overhead
  and small fixed costs dominate; the GA win is below noise.
- At `seq=2048` and beyond, the kernel is bandwidth-bound for K/V and the
  4× HBM-traffic reduction translates into a 14–17% wall-clock win.
- At `seq=4096`, the group-aware kernel beats PyTorch SDPA (which calls
  cuDNN's hand-tuned Flash Attention) at **0.87 ms vs 0.96 ms** (1.10×).
  This was not an explicit goal — beating cuDNN is rare for a Python-DSL
  kernel — but is a clean indication that the optimization is real.

### 5.4 Autotuned sweep

To rule out "default config is suboptimal for both" effects, we ran the
same sweep with `--tune`: both kernels autotune themselves over their
respective search spaces. (This is the apples-to-apples comparison after
section 3's fairness fix.)

| seq_len | Std cfg (tuned) | Standard | Group-aware | Official | sdpa | GA vs Std | GA vs Off |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 512  | (64,128,2,128) | 0.0290 | 0.0293 | 0.0290 | 0.0496 | 0.99x | 0.99x |
| 1024 | (64,128,2,128) | 0.0708 | 0.0711 | 0.0714 | 0.1010 | 1.00x | 1.00x |
| 2048 | (128,64,2,128) | 0.2241 | 0.2249 | 0.2206 | 0.2701 | 1.00x | 0.98x |
| 4096 | (64,128,2,128) | 0.8984 | **0.8377** | 0.8634 | 0.9597 | **1.07x** | **1.03x** |

Group-aware autotuned configs: `(16,128,2,128)` for seq ≤ 1024,
`(32,64,2,128)` at seq=2048, `(64,128,2,256)` at seq=4096.

This is *the* most important comparison in the project, and the result is
deeper than the default-config table suggests:

1. **Autotuning is a powerful equalizer.** At seq=512/1024/2048, both
   kernels are essentially tied once they tune. The 1.14–1.17× gap from
   section 5.3 collapses to ~1.00×. Autotuning the standard kernel raises
   its `block_M` from 64 → 128 at seq=2048, which doubles arithmetic
   intensity per gemm and reduces the loop-overhead share — partially
   compensating for the redundant KV loads.

2. **But it does not fully close the gap at long sequences.** At seq=4096
   the group-aware kernel is still 1.03x faster than the autotuned
   official and 1.07x faster than the autotuned standard. This is the
   residual algorithmic advantage that autotuning cannot find by itself,
   because no tile size can change the grid topology — the standard
   kernel's grid still launches `heads` blocks per (row block, batch),
   each loading its own K/V from HBM. The group-aware kernel restructures
   the grid; only that gets you below the redundant-load floor.

3. **The honest claim is "autotune + GA > autotune-only at long
   sequences."** It is not "GA crushes the official 1.17x." The latter is
   a default-config artifact and would be misleading in a report.

### 5.5 The two regimes: default-config vs autotuned

Both tables (5.3 default, 5.4 tuned) are useful, but they answer different
questions:

| Question | Use which table |
|---|---|
| *If I copy the official's defaults and don't tune, how much does GA help?* | 5.3 — answer: a lot (1.14–1.17x at seq ≥ 2048) |
| *If both kernels are properly tuned, is GA still better?* | 5.4 — answer: only at long sequences, and the gap is small (~1.03x at seq=4096) |
| *In practice with cold caches / first-run latency?* | 5.3 closer (most users don't tune) |
| *In a production setting with stable shapes?* | 5.4 (you tune once and amortize) |

A practitioner serving Mistral-style models at 4K–8K context with stable
shapes should tune, and will see a ~3% improvement from group-awareness on
top of tuning. A practitioner with variable shapes who relies on defaults
will see a much larger improvement (~15%).

### 5.6 Where does the win come from? (analysis)

We can decompose the GA-vs-Standard gap into a per-iteration view. Per
KV iteration, the standard kernel loads `block_N * dim * 2` bytes of K
(and another `block_N * dim * 2` for V). Total K HBM bytes across the
grid:

```
standard:  (seq_q/block_M) × heads × batch × (seq_kv/block_N) × block_N × dim × 2
        =  seq_q × heads × seq_kv × dim × 2 / block_M

group-aware: (seq_q/block_M) × kv_heads × batch × (seq_kv/block_N) × block_N × dim × 2
          =  seq_q × kv_heads × seq_kv × dim × 2 / block_M
```

Ratio: `heads / kv_heads = num_groups = 4`. So K HBM traffic is exactly
1/4 in the group-aware kernel (and same for V).

For `seq=4096, dim=64, fp16` and our shape:

- K bytes (standard): `4096 × 32 × 4096 × 64 × 2 / 64 = 1.07 GiB`
- K bytes (group-aware): `0.27 GiB`
- Same for V → total saving: `1.61 GiB` per call

A800 PCIe HBM2e has ~1.5 TB/s peak bandwidth. The 1.6 GiB saving corresponds
to ~1.1 ms at peak — which is more than the entire kernel runtime. Of course
real HBM efficiency is well below peak (~50–70% for streaming loads), and
Q/output traffic doesn't shrink, but the 100+ μs measured win at `seq=4096`
(0.99 → 0.87 ms) is squarely within the budget the bandwidth model predicts.

### 5.7 Where does it *not* help?

At `seq=512`, the kernel runtime is ~30 μs. Kernel-launch overhead, the
fixed cost of compiling the prologue (Q load, fill, max init), and the
software-pipeline drain at the end of the KV loop sum to a non-trivial
fraction of this. The bandwidth saving from KV reuse, while still 4×, is
saving 4× of a small absolute number. The standard and group-aware
kernels are within 1% — both effectively at the kernel-overhead floor.

This is why the *headline* of GQA group-awareness is **long-context
inference**, not short-context training. For LLMs serving 4K+ contexts
(which is the production setting), the optimization is fully active.

---

## 6. Discussion

### 6.1 Why doesn't the standard kernel hit this?

It could, in principle, hit it via shared-memory KV reuse across
*blocks*, if the GPU scheduler placed multiple Q-head blocks for the
same KV head on the same SM and L1/L2 reuse caught the second load. In
practice this is unreliable: nothing in the standard grid topology
*forces* same-group blocks onto the same SM, and L2 reuse is a soft
effect that depends on launch order, residency, and concurrent kernel
traffic. The group-aware kernel converts this from a "maybe L2 catches
it" optimization into an *explicit, deterministic* shared-memory reuse.

There is also a *partial* recovery via autotuning that we observed
empirically (section 5.4): when the standard kernel is allowed to tune,
it picks larger `block_M` (e.g., 128 at seq=2048), which reduces the
loop overhead and improves arithmetic intensity per gemm. This raises
the standard kernel's TFLOPS but does not change the total HBM bytes
it issues for K and V. So autotuning closes most of the gap at
`seq ≤ 2048` (where the kernel is compute-balanced) but leaves a
residual ~3% at `seq=4096` (where it becomes bandwidth-bound and the
4× HBM-traffic floor binds).

### 6.2 Why doesn't autotuning find the group-aware structure?

The autotune search space is a parameter sweep over tile sizes and
pipeline depth; it never alters the *grid* (`bx`, `by`, `bz`) topology
or how shared memory is indexed. Switching from "Y dim = heads, KV
indexed by by//ng" to "Y dim = kv_heads, all groups packed into one
block" is a kernel-structural change, not a parameter change. No
amount of autotuning can find it. This is exactly why algorithmic
contributions are still worthwhile in a tuner-rich ecosystem: they
expand what the tuner can search *over*, rather than competing with
the tuner inside the existing space.

### 6.3 When would the group-aware kernel hurt?

For `num_groups = 1` (i.e., plain MHA where `heads == kv_heads`), the
group-aware kernel degenerates to the standard kernel with extra
indexing overhead and should be slightly slower. For very small
`num_groups` (`2`), the bandwidth saving (`2×`) may not pay for the
register-pressure cost. We did not test these cases; the kernel is
specifically aimed at the production GQA shapes (`num_groups ∈ {4, 8}`)
used by Mistral, Llama-3, Qwen, etc.

### 6.4 What this project did *not* attempt

- **Flash Decoding / Split-KV.** For low-batch, low-head-count decoding
  (`batch=1, heads=1`), the group-aware kernel still has limited
  parallelism. Splitting the KV dimension across blocks (with a second
  reduction pass) would help. Not implemented — out of scope for a
  forward-pass project.
- **TMA (Hopper).** TileLang has TMA support but our target is Ampere.
- **bf16.** We tested only fp16. bf16 should work mechanically but was
  not validated.
- **The backward pass.** GQA backward has additional complications
  (atomicAdd accumulation on the reduced KV gradient) that don't apply
  to forward.

### 6.5 The CUDA diff harness, generalized

The CUDA-source diff added in `main()` (lines ~730–766) was indispensable
in the autotune-fairness investigation. We strongly recommend it as a
default debugging tool when comparing two TileLang kernels that are
*supposed* to be equivalent. If `get_kernel_source()` byte-matches at the
same tile config and threads, then any wall-clock difference is at most
measurement noise — and you should look for the cause elsewhere (decorator
wrapping, autotune skipping, benchmark order, cache state).

---

## 7. Reproduction

### 7.1 Run a single shape

```bash
# Default config, dim=64, seq_len=2048
python kernel/gqa_attention/gqa_flash_attention.py --seq_len 2048

# With autotuning (both kernels search their own spaces)
python kernel/gqa_attention/gqa_flash_attention.py --seq_len 2048 --tune

# Causal mask
python kernel/gqa_attention/gqa_flash_attention.py --seq_len 2048 --is_causal
```

### 7.2 Run the full sweep

```bash
# Sweep across seq_len ∈ {512, 1024, 2048, 4096} at default config
python kernel/gqa_attention/gqa_flash_attention.py --sweep

# Same, with both kernels autotuned per shape
python kernel/gqa_attention/gqa_flash_attention.py --sweep --tune
```

The sweep prints a per-shape line and ends with a summary table.

### 7.3 Inspect the generated CUDA

`main()` writes both kernels' CUDA source under `/tmp/`:

```
/tmp/gqa_flash_attn_s{seq_len}_d{dim}.cu          # standard kernel
/tmp/gqa_flash_attn_official_s{seq_len}_d{dim}.cu # official
/tmp/gqa_flash_attn_diff_s{seq_len}_d{dim}.diff   # unified diff (only if non-identical)
```

### 7.4 Code organization

- `_build_gqa_prim_func` — the standard GQA prim_func body.
- `gqa_flash_attn`, `gqa_flash_attn_tuned` — direct and autotuned wrappers.
- `_build_gqa_group_aware_prim_func` — the group-aware variant.
- `gqa_flash_attn_group_aware`, `gqa_flash_attn_group_aware_tuned` — wrappers.
- `_get_gqa_configs`, `_get_gqa_group_aware_configs` — autotune search spaces.
- `ref_gqa` — PyTorch reference for correctness.
- `main`, `benchmark_sweep` — drivers.

---

## 8. Conclusion

Three findings, in order of how much they generalize:

1. **The kernel body itself is essentially canonical FA1.** Reordering
   instructions inside the FA loop (rescale-before-exp vs after, fused
   max/scale loops, etc.) does not move the needle by more than
   measurement noise. We chased a 1.10x "win" from this for the better
   part of a session and the CUDA-source diff eventually proved that
   identifier renames were the only real change. **A 5-minute
   `get_kernel_source()` diff is the right first move when comparing
   two TileLang kernels that should be equivalent.**

2. **Autotune is not free to set up correctly, and bypassing it
   silently produces phantom wins.** The official `@autotune` decorator
   skips when all tunables are pre-bound at the call site, with a
   single-line warning. Most users will not notice. The honest fix
   (omit tile kwargs so autotune fires) collapsed a 1.10x result to
   1.00x and reshaped what counted as a "real" win for the rest of the
   project. Honest baselines matter more than fancy ones.

3. **Group-awareness — load KV once per `kv_head` and reuse across the
   `num_groups` Q heads — is the algorithmic GQA optimization that
   autotuning cannot find.** It restructures the grid topology, not the
   tile shape. At default configs it delivers 1.14–1.17x over the
   official at seq ≥ 2048 (and even beats cuDNN SDPA at seq=4096).
   When both kernels are properly autotuned, the gap narrows to ~1.03x
   at seq=4096 — autotune partially compensates by selecting larger Q
   tiles, but the 4× HBM-traffic floor still pinches at long
   sequences. The win that survives full tuning is small but real,
   and it would not exist at all without the grid restructuring.

For production GQA inference at 4K+ context, the group-aware kernel
should be the default. For short-context training where shape stability
is high and autotuning runs cheaply once, the standard kernel with
autotune is within a few percent.

---

## Appendix A: Autotune search spaces

### Standard kernel — `_get_gqa_configs(dim)`

```
block_M ∈ {64, 128}
block_N ∈ {64, 128}
num_stages ∈ {1, 2, 3}
threads ∈ {128, 256}
```

Filtered by:
- `threads % 32 == 0`
- `block_M / warp_count` and `block_N / warp_count` divisible by 16
- `2 × dtype_bytes × dim × (block_M + block_N) ≤ 100 KB` (SMEM budget)

Produces 15 valid configs for `dim=64`, 9 for `dim=128`.

### Group-aware kernel — `_get_gqa_group_aware_configs(dim, heads, kv_heads)`

```
block_M ∈ {16, 32, 64}        # smaller because M_tile = num_groups × block_M
block_N ∈ {32, 64, 128}
num_stages ∈ {1, 2, 3}
threads ∈ {128, 256}
```

Filtered by:
- `M_tile = num_groups × block_M`
- Warp alignment uses `M_tile`, not `block_M`
- SMEM bound is `dtype_bytes × dim × (2 × M_tile + 2 × block_N)` — Q and O
  shared tiles scale with `M_tile`

The search space adapts to `num_groups`, so the same code works for
Llama-2 (`num_groups=8`) and Mistral (`num_groups=4`).
