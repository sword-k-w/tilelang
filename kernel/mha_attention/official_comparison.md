# Official MHA FlashAttention Comparison

This note compares the five official MHA FlashAttention programs under
`examples/flash_attention`:

- `example_mha_fwd_bhsd.py`
- `example_mha_fwd_bshd.py`
- `example_mha_fwd_varlen.py`
- `example_mha_bwd_bhsd.py`
- `example_mha_bwd_bshd.py`

## Test Status

`kernel/mha_attention/official_comparison.py` was run with the default shape:

```text
batch=1, heads=16, seq_len=512, dim=64, causal=False, dtype=float16
backend=event, return_mode=mean
torch=2.6.0+cu124, cuda=12.4
gpu=NVIDIA A800 80GB PCIe
```

The run covered four dense methods:

- `fwd_bhsd`
- `fwd_bshd`
- `bwd_bhsd`
- `bwd_bshd`

`fwd_varlen` was not included because `official_comparison.py --methods all`
excludes it by default unless `--include-varlen` is passed. This is intentional:
the varlen path depends on the external `flash_attn` package for correctness and
reference timing.

All tested methods passed correctness:

- `fwd_bhsd`: output matched PyTorch reference.
- `fwd_bshd`: output matched PyTorch reference.
- `bwd_bhsd`: output, `dQ`, `dK`, and `dV` matched PyTorch autograd reference.
- `bwd_bshd`: output, `dQ`, `dK`, and `dV` matched PyTorch autograd reference.

## Performance Results

The log contains four result tables: one cold run with TileLang
compilation messages, followed by three warm runs. The three warm runs are the
main numbers below because they remove first-run compilation/cache effects.

### Warm Run Latency

| Method | Run 1 latency (ms) | Run 2 latency (ms) | Run 3 latency (ms) | Mean latency (ms) | Std dev (ms) | CV |
|---|---:|---:|---:|---:|---:|---:|
| `fwd_bhsd` | 0.0270 | 0.0268 | 0.0269 | 0.0269 | 0.0001 | 0.37% |
| `fwd_bshd` | 0.0269 | 0.0270 | 0.0270 | 0.0270 | 0.0001 | 0.21% |
| `bwd_bhsd` | 0.1428 | 0.1463 | 0.1435 | 0.1442 | 0.0019 | 1.28% |
| `bwd_bshd` | 0.1408 | 0.1451 | 0.1430 | 0.1430 | 0.0022 | 1.50% |

### Warm Run Throughput

| Method | Run 1 TFLOPS | Run 2 TFLOPS | Run 3 TFLOPS | Mean TFLOPS |
|---|---:|---:|---:|---:|
| `fwd_bhsd` | 39.805 | 40.019 | 39.949 | 39.924 |
| `fwd_bshd` | 39.850 | 39.750 | 39.833 | 39.811 |
| `bwd_bhsd` | 18.797 | 18.354 | 18.703 | 18.618 |
| `bwd_bshd` | 19.068 | 18.501 | 18.773 | 18.781 |

### Cold Run Note

The first run included TileLang compilation for all kernels. Its measured
latencies were:

| Method | Cold-run latency (ms) | Cold-run TFLOPS |
|---|---:|---:|
| `fwd_bhsd` | 0.0336 | 31.965 |
| `fwd_bshd` | 0.0273 | 39.362 |
| `bwd_bhsd` | 0.1814 | 14.799 |
| `bwd_bshd` | 0.1850 | 14.507 |

The cold-run backward numbers are about 26-29% slower than the warm-run means.
This is consistent with first-run effects from JIT compilation, kernel loading,
allocator/cache state, or other one-time setup. For reporting kernel runtime,
use the warm-run means.

## Result Analysis

Forward BHSD and BSHD are effectively tied for this shape. The warm means are
0.0269 ms and 0.0270 ms, only about 0.25% apart. This difference is below normal
benchmark noise, and both layouts deliver about 40 TFLOPS.

Backward is roughly 5.3x slower than forward at this shape:

- BHSD: `0.1442 / 0.0269 = 5.36x`
- BSHD: `0.1430 / 0.0270 = 5.30x`

This is expected because backward runs the forward-style attention reconstruction
plus gradient paths for `dQ`, `dK`, and `dV`, including atomic accumulation for
`dQ`.

For backward, BSHD is slightly faster than BHSD in the warm-run mean:

- `bwd_bhsd`: 0.1442 ms
- `bwd_bshd`: 0.1430 ms
- Difference: about 0.86%

That small gap is not large enough to claim a general layout advantage. The
source-level difference is mostly indexing/layout; with this problem size on an
A800, both dense layouts perform similarly.

## High-Level Differences

| Program | Direction | Tensor layout | Sequence model | Reference path | Main purpose |
|---|---|---|---|---|---|
| `example_mha_fwd_bhsd.py` | Forward | `[batch, heads, seq, dim]` | Dense, supports `seq_q <= seq_kv` | In-file PyTorch einsum + softmax | Forward MHA for BHSD layout, including decode-style unequal Q/KV lengths |
| `example_mha_fwd_bshd.py` | Forward | `[batch, seq, heads, dim]` | Dense, same Q/KV length | In-file PyTorch einsum + softmax | Forward MHA for BSHD layout |
| `example_mha_fwd_varlen.py` | Forward | Unpadded `[total_tokens, heads, dim]` plus cumulative sequence lengths | Variable length / padded batches | `flash_attn.flash_attn_varlen_func` | Forward MHA for packed variable-length batches |
| `example_mha_bwd_bhsd.py` | Forward + backward | `[batch, heads, seq, dim]` | Dense, same Q/KV length | In-file PyTorch autograd reference | Autograd-compatible backward MHA for BHSD layout |
| `example_mha_bwd_bshd.py` | Forward + backward | `[batch, seq, heads, dim]` | Dense, same Q/KV length | In-file PyTorch autograd reference | Autograd-compatible backward MHA for BSHD layout |

## Forward Dense Kernels

`example_mha_fwd_bhsd.py` and `example_mha_fwd_bshd.py` implement the same
FlashAttention forward algorithm:

1. Load a block of Q into shared memory.
2. Sweep K/V blocks.
3. Compute `Q @ K^T` into `acc_s`.
4. Apply causal or out-of-bounds masking.
5. Maintain online softmax state using `scores_max`, `scores_max_prev`,
   `scores_scale`, `scores_sum`, and `logsum`.
6. Accumulate `softmax(QK^T) @ V` into `acc_o`.
7. Normalize by `logsum` and write the output.

The important difference is layout:

- BHSD uses `Q[b, h, q, d]`, `K[b, h, k, d]`, and output `O[b, h, q, d]`.
- BSHD uses `Q[b, q, h, d]`, `K[b, k, h, d]`, and output `O[b, q, h, d]`.

The BHSD forward version is also more general for sequence lengths. It accepts
separate `seq_q` and `seq_kv` and computes `past_len = seq_kv - seq_q`, so the
causal mask is right-aligned for decode/prefix-cache style cases. The BSHD
version only takes one `seq_len`, so Q, K, and V have the same sequence length.

## Variable-Length Forward Kernel

`example_mha_fwd_varlen.py` targets padded batches that have been packed into
unpadded token tensors. Instead of dense `[batch, seq, heads, dim]` storage, it
uses:

- `Q_unpad`, `K_unpad`, `V_unpad` with shape `[total_tokens, heads, dim]`.
- `cu_seqlens_q` and `cu_seqlens_k` to recover each batch item's token range.
- `max_seqlen_q` to define the grid size.

Inside the kernel, `bz` selects the batch item, and the actual Q/KV start/end
positions are loaded from cumulative sequence length arrays. The masking logic
must handle both causal constraints and per-example out-of-bounds positions.

This version compares against the external FlashAttention package via
`flash_attn.flash_attn_varlen_func`, so it has an extra dependency beyond
PyTorch and TileLang.

## Backward Kernels

`example_mha_bwd_bhsd.py` and `example_mha_bwd_bshd.py` are layout variants of
the same backward design. Each file contains multiple TileLang kernels:

- `flashattn_fwd`: computes forward output and log-sum-exp (`lse`) for backward.
- `flashattn_bwd_preprocess`: computes `Delta = sum(O * dO)` per row.
- `flashattn_bwd`: computes `dQ`, `dK`, and `dV`.
- `flashattn_bwd_postprocess`: converts the accumulated float32 `dQ` buffer to
  the final float16 output layout.

The main backward kernel iterates over K/V blocks, reconstructs the attention
probabilities from `QK^T` and `lse`, forms the softmax derivative term, and then
updates:

- `dV` from `P^T @ dO`
- `dK` from `dS^T @ Q`
- `dQ` from `dS @ K`

`dQ` uses `T.atomic_add` because multiple K/V blocks contribute to the same Q
gradient rows. Both backward variants define a custom `make_dq_layout` so the
atomic-add buffer matches the expected GEMM fragment layout before the
postprocess copy.

The difference between the two backward files is again layout:

- BHSD indexes tensors as `[batch, heads, seq, dim]`.
- BSHD indexes tensors as `[batch, seq, heads, dim]`.

## Practical Selection

- Use `example_mha_fwd_bhsd.py` if the surrounding model/runtime uses BHSD or
  if Q and KV lengths may differ.
- Use `example_mha_fwd_bshd.py` if the surrounding code uses BSHD and dense
  same-length Q/K/V tensors.
- Use `example_mha_fwd_varlen.py` for padded batches where packed variable-length
  execution avoids wasted work on padding tokens.
- Use `example_mha_bwd_bhsd.py` or `example_mha_bwd_bshd.py` when gradients are
  required; choose the file that matches the tensor layout used by the caller.
