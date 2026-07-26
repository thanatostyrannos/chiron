"""Measure device-copy bandwidth as a function of GPU working-set footprint.

Why this exists: on gfx1151 the ROCm pool (~83 GiB) is far larger than the BIOS UMA
dedicated carve-out (16 GiB by default), because the driver reaches into GTT/shared
system memory. Capacity that is 2x slower is a *different tier*, not more of the same
tier -- and for a memory-systems lab that distinction sizes every KV-cache experiment.

This probe finds where the fast tier ends. Run it before and after any BIOS UMA FB
Size change; the two curves are the experiment (see
notebook/uma-carveout-controls-fast-tier.md).

Reports total footprint (2 buffers) rather than buffer size, because footprint is what
the allocator must place.

Usage:
    python scripts/measure_memory_bandwidth_tiers.py [--coarse | --fine]
"""

from __future__ import annotations

import argparse
import json
import time

import torch

COARSE_BUF_GIB = [1, 2, 4, 8, 12, 20, 30, 36]
FINE_BUF_GIB = [13, 14, 15, 16, 17, 18, 19, 20]

# A footprint counts as "fast tier" at or above this measured bandwidth. Set from the
# 2026-07-26 control run, where the fast plateau sat at 184-210 GB/s and the slow
# region at 61-114 GB/s -- the gap is wide enough that the exact cut is not delicate.
FAST_TIER_FLOOR_GB_S = 170.0


def measure_copy_bandwidth_gb_s(buffer_gib: float, iters: int = 3) -> float | None:
    """Return device-to-device copy bandwidth in GB/s, or None if allocation fails."""
    n = int(buffer_gib * 1024**3 // 2)  # fp16 = 2 bytes/elem
    try:
        src = torch.empty(n, dtype=torch.float16, device="cuda")
        dst = torch.empty(n, dtype=torch.float16, device="cuda")
    except (RuntimeError, torch.OutOfMemoryError):
        torch.cuda.empty_cache()
        return None

    try:
        src.fill_(1.0)
        torch.cuda.synchronize()
        for _ in range(2):  # warmup
            dst.copy_(src)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            dst.copy_(src)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / iters
    finally:
        del src, dst
        torch.cuda.empty_cache()

    moved_bytes = 2 * n * 2  # one read + one write, 2 bytes/elem
    return moved_bytes / elapsed / 1e9


def fast_tier_boundary_gib(results: list[tuple[float, float]]) -> float:
    """Largest measured footprint still at or above the fast-tier floor."""
    fast = [footprint for footprint, bw in results if bw >= FAST_TIER_FLOOR_GB_S]
    return max(fast) if fast else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--coarse", action="store_true", help="wide sweep (default)")
    group.add_argument("--fine", action="store_true", help="narrow sweep around the cliff")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible -- activate the lab venv (scripts/activate-lab.ps1)")

    props = torch.cuda.get_device_properties(0)
    print(f"device: {torch.cuda.get_device_name(0)}  arch: {getattr(props, 'gcnArchName', '?')}")
    print(f"torch: {torch.__version__}  reported_pool_GiB: {props.total_memory / 1024**3:.2f}")

    sizes = FINE_BUF_GIB if args.fine else COARSE_BUF_GIB
    results: list[tuple[float, float]] = []

    print(f"\n{'buf_GiB':>8} {'footprint_GiB':>14} {'GB/s':>8}")
    for gib in sizes:
        bandwidth = measure_copy_bandwidth_gb_s(gib)
        if bandwidth is None:
            print(f"{gib:>8} {2 * gib:>14} {'ALLOC FAIL':>8}")
            break
        results.append((2.0 * gib, bandwidth))
        print(f"{gib:>8} {2 * gib:>14} {bandwidth:>8.1f}")

    boundary = fast_tier_boundary_gib(results)
    print(f"\nfast_tier_boundary_GiB (>={FAST_TIER_FLOOR_GB_S:.0f} GB/s): {boundary:.0f}")
    print(json.dumps({"fast_tier_boundary_gib": boundary, "samples": results}))


if __name__ == "__main__":
    main()
