#!/usr/bin/env python3
"""Benchmark the official MHA FlashAttention examples.

Run from the repository root, for example:

    python kernel/mha_attention/official_comparison.py --methods all --batch 1 --heads 16 --seq-len 512 --dim 64

The script imports the official programs under examples/flash_attention and
benchmarks their TileLang kernels plus the available reference methods.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
FLASH_ATTENTION_DIR = REPO_ROOT / "examples" / "flash_attention"

METHODS = ("fwd_bhsd", "fwd_bshd", "fwd_varlen", "bwd_bhsd", "bwd_bshd")


@dataclass
class BenchResult:
    method: str
    target: str
    status: str
    latency_ms: float | None
    tflops: float | None
    message: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        choices=["all", *METHODS],
        help="Methods to benchmark.",
    )
    parser.add_argument("--batch", type=int, default=1, help="Batch size.")
    parser.add_argument("--heads", type=int, default=16, help="Number of attention heads.")
    parser.add_argument("--seq-len", type=int, default=512, help="Dense sequence length and varlen max sequence length.")
    parser.add_argument("--seq-q", type=int, default=None, help="BHSD forward query length. Defaults to --seq-len.")
    parser.add_argument("--seq-kv", type=int, default=None, help="BHSD forward key/value length. Defaults to --seq-len.")
    parser.add_argument("--dim", type=int, default=64, help="Head dimension.")
    parser.add_argument("--causal", action="store_true", help="Use causal attention.")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16", help="Input dtype.")
    parser.add_argument("--warmup", type=float, default=25, help="Warmup time in ms for do_bench.")
    parser.add_argument("--rep", type=float, default=100, help="Benchmark time in ms for do_bench.")
    parser.add_argument("--n-warmup", type=int, default=0, help="Manual warmup iterations for do_bench.")
    parser.add_argument("--n-repeat", type=int, default=0, help="Manual repeat iterations for do_bench.")
    parser.add_argument("--backend", choices=["event", "cupti", "cudagraph"], default="event", help="TileLang do_bench backend.")
    parser.add_argument("--return-mode", choices=["mean", "median", "min", "max"], default="mean", help="do_bench aggregation mode.")
    parser.add_argument("--skip-correctness", action="store_true", help="Skip correctness checks before timing.")
    parser.add_argument("--include-reference", action="store_true", help="Benchmark PyTorch / FlashAttention references too.")
    parser.add_argument("--include-varlen", action="store_true", help="Include fwd_varlen when --methods all is used.")
    parser.add_argument("--device", default="cuda", help="Torch device, normally cuda or cuda:0.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def selected_methods(args: argparse.Namespace) -> list[str]:
    if "all" not in args.methods:
        return list(dict.fromkeys(args.methods))
    methods = ["fwd_bhsd", "fwd_bshd", "bwd_bhsd", "bwd_bshd"]
    if args.include_varlen:
        methods.append("fwd_varlen")
    return methods


def ensure_import_path() -> None:
    for path in (REPO_ROOT, FLASH_ATTENTION_DIR):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def import_runtime_modules():
    ensure_import_path()
    try:
        import torch
        from tilelang.profiler import do_bench
    except Exception as exc:
        raise RuntimeError(
            "Failed to import runtime dependencies. Build/install TileLang and install torch before running this benchmark."
        ) from exc

    return torch, do_bench


def import_example_module(name: str):
    ensure_import_path()
    return importlib.import_module(name)


def dtype_from_name(torch, name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def bench_call(
    do_bench: Callable,
    fn: Callable,
    args: argparse.Namespace,
) -> float:
    return float(
        do_bench(
            fn,
            warmup=args.warmup,
            rep=args.rep,
            _n_warmup=args.n_warmup,
            _n_repeat=args.n_repeat,
            backend=args.backend,
            return_mode=args.return_mode,
            device=args.device,
        )
    )


def dense_forward_flops(batch: int, heads: int, seq_q: int, seq_kv: int, dim: int, causal: bool) -> float:
    flops = 4.0 * batch * heads * seq_q * seq_kv * dim
    return flops * 0.5 if causal and seq_q == seq_kv else flops


def dense_backward_flops(batch: int, heads: int, seq_len: int, dim: int, causal: bool) -> float:
    flops = 10.0 * batch * heads * seq_len * seq_len * dim
    return flops * 0.5 if causal else flops


def tflops(total_flops: float, latency_ms: float) -> float:
    return total_flops / latency_ms * 1e-9


def assert_close(torch, actual, expected, *, name: str, rtol: float = 1e-2, atol: float = 1e-2) -> None:
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    print(f"  correctness: {name} passed")


def run_fwd_bhsd(args: argparse.Namespace, torch, do_bench: Callable) -> list[BenchResult]:
    mod = import_example_module("example_mha_fwd_bhsd")
    dtype = dtype_from_name(torch, args.dtype)
    seq_q = args.seq_q or args.seq_len
    seq_kv = args.seq_kv or args.seq_len
    if seq_kv < seq_q:
        raise ValueError("fwd_bhsd requires seq_kv >= seq_q")

    torch.manual_seed(args.seed)
    q = torch.randn(args.batch, args.heads, seq_q, args.dim, device=args.device, dtype=dtype)
    k = torch.randn(args.batch, args.heads, seq_kv, args.dim, device=args.device, dtype=dtype)
    v = torch.randn(args.batch, args.heads, seq_kv, args.dim, device=args.device, dtype=dtype)

    kernel = mod.flashattn(
        args.batch,
        args.heads,
        seq_q,
        seq_kv,
        args.dim,
        args.causal,
        block_M=64,
        block_N=64,
        num_stages=1,
        threads=128,
    )
    ref = partial(mod.ref_program, is_causal=args.causal)
    if not args.skip_correctness:
        assert_close(torch, kernel(q, k, v), ref(q, k, v), name="fwd_bhsd")

    total_flops = dense_forward_flops(args.batch, args.heads, seq_q, seq_kv, args.dim, args.causal)
    results = []
    latency = bench_call(do_bench, lambda: kernel(q, k, v), args)
    results.append(BenchResult("fwd_bhsd", "tilelang", "ok", latency, tflops(total_flops, latency)))
    if args.include_reference:
        latency = bench_call(do_bench, lambda: ref(q, k, v), args)
        results.append(BenchResult("fwd_bhsd", "torch_ref", "ok", latency, tflops(total_flops, latency)))
    return results


def run_fwd_bshd(args: argparse.Namespace, torch, do_bench: Callable) -> list[BenchResult]:
    mod = import_example_module("example_mha_fwd_bshd")
    dtype = dtype_from_name(torch, args.dtype)

    torch.manual_seed(args.seed)
    q = torch.randn(args.batch, args.seq_len, args.heads, args.dim, device=args.device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    kernel = mod.flashattn(
        args.batch,
        args.heads,
        args.seq_len,
        args.dim,
        args.causal,
        block_M=128,
        block_N=128,
        num_stages=1,
        threads=128,
    )
    ref = partial(mod.ref_program, is_causal=args.causal)
    if not args.skip_correctness:
        assert_close(torch, kernel(q, k, v), ref(q, k, v), name="fwd_bshd")

    total_flops = dense_forward_flops(args.batch, args.heads, args.seq_len, args.seq_len, args.dim, args.causal)
    results = []
    latency = bench_call(do_bench, lambda: kernel(q, k, v), args)
    results.append(BenchResult("fwd_bshd", "tilelang", "ok", latency, tflops(total_flops, latency)))
    if args.include_reference:
        latency = bench_call(do_bench, lambda: ref(q, k, v), args)
        results.append(BenchResult("fwd_bshd", "torch_ref", "ok", latency, tflops(total_flops, latency)))
    return results


def run_fwd_varlen(args: argparse.Namespace, torch, do_bench: Callable) -> list[BenchResult]:
    mod = import_example_module("example_mha_fwd_varlen")
    try:
        import flash_attn
        from varlen_utils import generate_qkv, generate_random_padding_mask
    except Exception as exc:
        return [BenchResult("fwd_varlen", "tilelang", "skipped", None, None, f"missing varlen dependency: {exc}")]

    dtype = dtype_from_name(torch, args.dtype)
    torch.manual_seed(args.seed)
    q = torch.randn(args.batch, args.seq_len, args.heads, args.dim, dtype=dtype, device=args.device, requires_grad=True)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    query_padding_mask = generate_random_padding_mask(args.seq_len, args.batch, args.device, mode="random")
    key_padding_mask = generate_random_padding_mask(args.seq_len, args.batch, args.device, mode="random")
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        _q_pad,
        _k_pad,
        _v_pad,
        output_pad_fn,
        _dq_pad_fn,
        _dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)

    kernel = mod.flashattn(
        args.batch,
        q_unpad.shape[0],
        k_unpad.shape[0],
        args.heads,
        args.dim,
        args.causal,
        block_M=64,
        block_N=64,
        num_stages=1,
        threads=128,
    )

    def run_tilelang():
        return kernel(q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q)

    def run_flash_attn():
        return flash_attn.flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            0.0,
            causal=args.causal,
        )

    if not args.skip_correctness:
        assert_close(torch, output_pad_fn(run_tilelang()), output_pad_fn(run_flash_attn()), name="fwd_varlen")

    # Match the official example's FLOP estimate. Padding randomness means this
    # is based on max seq len, not exact unpadded token pairs.
    total_flops = dense_forward_flops(args.batch, args.heads, args.seq_len, args.seq_len, args.dim, args.causal)
    results = []
    latency = bench_call(do_bench, run_tilelang, args)
    results.append(BenchResult("fwd_varlen", "tilelang", "ok", latency, tflops(total_flops, latency)))
    if args.include_reference:
        latency = bench_call(do_bench, run_flash_attn, args)
        results.append(BenchResult("fwd_varlen", "flash_attn_ref", "ok", latency, tflops(total_flops, latency)))
    return results


def run_bwd(
    args: argparse.Namespace,
    torch,
    do_bench: Callable,
    *,
    method: str,
    module_name: str,
    layout: str,
) -> list[BenchResult]:
    mod = import_example_module(module_name)
    dtype = dtype_from_name(torch, args.dtype)
    torch.manual_seed(args.seed)

    if layout == "bhsd":
        shape = (args.batch, args.heads, args.seq_len, args.dim)
    elif layout == "bshd":
        shape = (args.batch, args.seq_len, args.heads, args.dim)
    else:
        raise ValueError(f"unsupported layout: {layout}")

    q = torch.empty(shape, dtype=dtype, device=args.device).normal_().requires_grad_()
    k = torch.empty_like(q).normal_().requires_grad_()
    v = torch.empty_like(q).normal_().requires_grad_()
    do = torch.randn_like(q)

    def run_tilelang_backward():
        for tensor in (q, k, v):
            tensor.grad = None
        out = mod.attention(q, k, v, args.causal)
        out.backward(do)
        return out

    def run_torch_backward():
        for tensor in (q, k, v):
            tensor.grad = None
        out = mod.ref_program(q, k, v, args.causal)
        out.backward(do)
        return out

    if not args.skip_correctness:
        tile_out = run_tilelang_backward()
        tile_grads = (q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone())
        torch_out = run_torch_backward()
        torch_grads = (q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone())
        assert_close(torch, tile_out, torch_out, name=f"{method}.output")
        for grad_name, actual, expected in zip(("dQ", "dK", "dV"), tile_grads, torch_grads):
            assert_close(torch, actual, expected, name=f"{method}.{grad_name}")

    total_flops = dense_backward_flops(args.batch, args.heads, args.seq_len, args.dim, args.causal)
    results = []
    latency = bench_call(do_bench, run_tilelang_backward, args)
    results.append(BenchResult(method, "tilelang_autograd", "ok", latency, tflops(total_flops, latency)))
    if args.include_reference:
        latency = bench_call(do_bench, run_torch_backward, args)
        results.append(BenchResult(method, "torch_autograd", "ok", latency, tflops(total_flops, latency)))
    return results


def run_method(method: str, args: argparse.Namespace, torch, do_bench: Callable) -> list[BenchResult]:
    if method == "fwd_bhsd":
        return run_fwd_bhsd(args, torch, do_bench)
    if method == "fwd_bshd":
        return run_fwd_bshd(args, torch, do_bench)
    if method == "fwd_varlen":
        return run_fwd_varlen(args, torch, do_bench)
    if method == "bwd_bhsd":
        return run_bwd(args, torch, do_bench, method=method, module_name="example_mha_bwd_bhsd", layout="bhsd")
    if method == "bwd_bshd":
        return run_bwd(args, torch, do_bench, method=method, module_name="example_mha_bwd_bshd", layout="bshd")
    raise ValueError(f"unknown method: {method}")


def print_environment(args: argparse.Namespace, torch) -> None:
    print("Benchmark configuration:")
    print(f"  repo_root={REPO_ROOT}")
    print(f"  methods={', '.join(selected_methods(args))}")
    print(f"  shape=batch={args.batch}, heads={args.heads}, seq_len={args.seq_len}, dim={args.dim}, causal={args.causal}")
    print(f"  dtype={args.dtype}, device={args.device}, backend={args.backend}, return_mode={args.return_mode}")
    print(f"  torch={torch.__version__}, cuda={torch.version.cuda}, cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device = torch.device(args.device)
        index = torch.cuda.current_device() if device.index is None else device.index
        print(f"  gpu={torch.cuda.get_device_name(index)}")


def print_results(results: list[BenchResult]) -> None:
    headers = ("method", "target", "status", "latency_ms", "tflops", "message")
    rows = []
    for result in results:
        rows.append(
            (
                result.method,
                result.target,
                result.status,
                "" if result.latency_ms is None else f"{result.latency_ms:.4f}",
                "" if result.tflops is None or math.isnan(result.tflops) else f"{result.tflops:.3f}",
                result.message,
            )
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt(row):
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    print()
    print(fmt(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(fmt(row))


def write_csv(path: Path, results: list[BenchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    print(f"Wrote CSV: {path}")


def write_json(path: Path, args: argparse.Namespace, results: list[BenchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "config": vars(args),
        "results": [asdict(result) for result in results],
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    print(f"Wrote JSON: {path}")


def main() -> int:
    args = parse_args()
    torch, do_bench = import_runtime_modules()
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is not available, but this benchmark needs a CUDA device.")

    print_environment(args, torch)
    results: list[BenchResult] = []
    for method in selected_methods(args):
        print(f"\n== {method} ==")
        try:
            method_results = run_method(method, args, torch, do_bench)
        except Exception as exc:
            method_results = [BenchResult(method, "all", "failed", None, None, repr(exc))]
            print(f"  failed: {exc!r}")
        results.extend(method_results)

    print_results(results)
    if args.csv and results:
        write_csv(args.csv, results)
    if args.json and results:
        write_json(args.json, args, results)
    failed = any(result.status == "failed" for result in results)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
