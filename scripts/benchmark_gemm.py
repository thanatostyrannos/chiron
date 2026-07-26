"""GEMM throughput at 8192^3 in bf16 and fp16.

Guards against the documented hipBLASLt misconfiguration that drops GEMM ~5x
(ROCm #6022: ~33 TFLOPS correctly configured vs ~6 TFLOPS with a bad Tensile library
path). The configuration is validated by the resulting number, never by the path
merely existing on disk.

Usage:
    scripts/activate-lab.ps1        # sets HIPBLASLT_TENSILE_LIBPATH
    python scripts/benchmark_gemm.py
"""

from __future__ import annotations

import json
import os
import time

import torch

MATRIX_DIM = 8192
WARMUP_ITERS = 3
TIMED_ITERS = 10


def gemm_tflops(dim: int, dtype: torch.dtype) -> float:
    a = torch.randn(dim, dim, device="cuda", dtype=dtype)
    b = torch.randn(dim, dim, device="cuda", dtype=dtype)
    try:
        for _ in range(WARMUP_ITERS):
            a @ b
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(TIMED_ITERS):
            a @ b
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / TIMED_ITERS
    finally:
        del a, b
        torch.cuda.empty_cache()

    return (2 * dim**3) / elapsed / 1e12


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible -- activate the lab venv (scripts/activate-lab.ps1)")

    print("HIPBLASLT_TENSILE_LIBPATH:", os.environ.get("HIPBLASLT_TENSILE_LIBPATH", "<unset>"))
    print("TORCH_BLAS_PREFER_HIPBLASLT:", os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT", "<unset>"))

    measured = {}
    for dtype in (torch.bfloat16, torch.float16):
        name = str(dtype).rsplit(".", maxsplit=1)[-1]
        tflops = gemm_tflops(MATRIX_DIM, dtype)
        measured[name] = round(tflops, 1)
        print(f"{name}: {tflops:.1f} TFLOPS")

    print(json.dumps({"matrix_dim": MATRIX_DIM, "tflops": measured}))


if __name__ == "__main__":
    main()
