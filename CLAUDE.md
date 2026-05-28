# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TileLang is a Python DSL for writing high-performance GPU/CPU kernels (GEMM, FlashAttention, convolutions, etc.). It compiles tile-level operations into optimized CUDA, HIP, Metal, or C code via Apache TVM. The repo builds a native C++ extension (`libtilelang.so`) linked against a bundled TVM fork, plus optional CUTLASS and Composable Kernel submodules.

## Build & Develop

```bash
# Editable install (after setting up venv and pre-commit)
pip install --no-build-isolation --verbose --editable .

# Build with CMake directly (for C++ iteration)
cmake -S . -B build -DUSE_CUDA=ON
cmake --build build
```

## Test

```bash
# Run all tests (from repo root)
pytest testing

# Run a single test file
pytest testing/python/kernel/test_gemm.py

# Run a specific test by name
pytest testing/python/kernel/test_gemm.py -k "test_gemm_float16"

# Run tests in parallel (as CI does)
pytest --numprocesses=8 testing

# Skip perf/slow tests (default behavior; these are gated behind markers)
pytest testing -m "not perf and not slow"

# Include performance benchmarks
pytest testing --run-perf
```

## Lint

```bash
# Run all pre-commit checks
pre-commit run --all-files

# Or use the format script (runs on changed files vs upstream main)
bash format.sh
```

Pre-commit runs: ruff (lint + format), clang-format (C++), codespell, pymarkdown.

## Architecture

### DSL Layer (`tilelang/language/`)

Users write kernels with two styles, both entry-pointed through `@tilelang.jit`:

- **Eager mode**: Annotations on parameters declare tensor shapes/dtypes. `T.Kernel`, `T.copy`, `T.gemm`, etc. build a `PrimFunc` implicitly. Calling the decorated function compiles *and* executes immediately.
- **Lazy mode**: The function explicitly returns a `T.prim_func`. Calling it returns a compiled `JITKernel` object for manual invocation.

Core DSL primitives: `T.Kernel` (launch grid), `T.alloc_shared`/`T.alloc_fragment` (memory allocation), `T.copy`/`T.gemm`/`T.reduce` (tile ops), `T.Pipelined`/`T.Parallel` (loop annotations), `T.annotate_layout`/`T.use_swizzle` (layout hints).

### Compilation Pipeline (`tilelang/engine/lower.py`)

```
PrimFunc → resolve_pipeline(target) → pipeline.lower(mod, target) → split host/device → codegen
```

1. **`resolve_pipeline`** (`tilelang/backend/pipeline.py`) selects a backend-specific `Pipeline` by `target.kind.name` (cuda/hip/metal/c/llvm).
2. The pipeline runs a sequence of TIR transform passes (defined in `tilelang/transform/` and `tilelang/{backend}/transform/`).
3. `lower_to_host_device_ir` splits the module into host and device IRModules via `tirx.transform.Filter`.
4. `device_codegen` or `device_codegen_without_compile` dispatches to target-specific codegen (CUDA → nvcc, HIP → hipcc, Metal → Metal, C → C compiler).

### Backend Pipeline Registration

Each backend registers a `Pipeline` in its own module:
- `tilelang/backend/cuda/pipeline.py` — `CUDAPassPipelineBody`
- `tilelang/backend/rocm/pipeline.py` — HIP pipeline
- `tilelang/backend/metal/pipeline.py` — Metal pipeline
- `tilelang/backend/cpu/pipeline.py` — CPU pipeline (also reused by webgpu via `tilelang/backend/common.py`)

### JIT & Caching (`tilelang/jit/`)

`JITImpl` (created by `@tilelang.jit`) manages compilation caching and mode inference. The `compile()` free function orchestrates `engine.lower()` → `JITKernel` wrapping. `par_compile()` compiles multiple configs in parallel via thread pool.

### C++ Extension (`src/`)

The native library provides:
- **Codegen**: `src/backend/{cuda,rocm,metal,webgpu,cpu}/` — Target-specific code generation
- **Runtime**: `src/runtime/` — Execution support
- **Transforms**: `src/transform/` — TIR passes implemented in C++ for performance
- **Tile ops**: `src/op/` — Tile-level operation implementations
- **Bindings**: Python ↔ C++ via TVM's FFI (`_ffi_api`)

### Key Submodules (in `3rdparty/`)

- `tvm` — TileLang's fork of Apache TVM (TIR infrastructure, codegen, runtime)
- `cutlass` — NVIDIA CUTLASS templates
- `composable_kernel` — AMD Composable Kernel library

## Environment Variables

| Variable | Purpose |
|---|---|
| `TILELANG_TARGET` | Default compilation target (`cuda`, `llvm`, `auto`) |
| `TILELANG_EXECUTION_BACKEND` | Kernel execution backend (`dlpack`, `tvm_ffi`, `cython`, `auto`) |
| `TILELANG_VERBOSE` | Enable verbose compilation output |
| `SKIP_LOADING_TILELANG_SO` | Set to `"1"` to skip loading the native library |

## Code Style

- Python: ruff format/docstring, line length 140, double quotes, 4-space indent
- C++: clang-format LLVM style, 2-space indent, 80-column limit, C++17
- Pre-commit hooks enforce both; install them with `pre-commit install --install-hooks`
