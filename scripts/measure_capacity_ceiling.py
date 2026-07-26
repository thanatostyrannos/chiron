"""Measure the largest GPU allocation that can actually be written and read back.

This is the number that bounds every long-context / KV-capacity experiment in the lab
(ASSUMPTIONS.md: hardware-capacity-ceiling). Three traps this probe exists to avoid:

1. **The reported pool is not the ceiling.** torch reported 82.99 GiB on this unit
   while allocations beyond that still succeeded -- the driver oversubscribes into
   system memory.
2. **A search bound is not a measurement.** An earlier version capped its search at
   the reported pool and "measured" exactly the cap; `scripts/preflight.ps1` bounds at
   100 and reported exactly 100. A saturated search reports its own bound, so this
   probe says so explicitly rather than returning a number that looks measured.
3. **An untouched allocation proves nothing.** `torch.empty` can hand back a virtual
   reservation that fails on first write. Every trial here fills the buffer and reads
   a sample back, so the reported ceiling is memory that actually works.

Re-run after any BIOS UMA, ROCm, or driver change -- an upgrade is a change of
instrument.

Usage:
    python scripts/measure_capacity_ceiling.py [--max-gib 104]
"""

from __future__ import annotations

import argparse
import json
import time

import torch

RESOLUTION_GIB = 1.0

# Default search bound. Deliberately below installed RAM: filling a buffer larger than
# physical memory drives the host into swap, and a hung workstation is a worse outcome
# than an unmeasured tail. If the search saturates here, the result says so.
DEFAULT_MAX_GIB = 104.0


def allocation_survives_write(gib: float) -> bool:
    """Allocate `gib` GiB, write every byte, read a sample back. True if all succeed."""
    n = int(gib * 1024**3 // 2)  # fp16 = 2 bytes/elem
    buf = None
    try:
        buf = torch.empty(n, dtype=torch.float16, device="cuda")
        buf.fill_(1.0)
        torch.cuda.synchronize()
        return buf[:: max(1, n // 1024)].sum().item() != 0
    except (RuntimeError, torch.OutOfMemoryError):
        return False
    finally:
        del buf
        torch.cuda.empty_cache()


def max_writable_allocation_gib(upper_bound_gib: float) -> tuple[float, bool]:
    """Binary-search the largest writable allocation. Returns (gib, search_saturated)."""
    if allocation_survives_write(upper_bound_gib):
        return upper_bound_gib, True

    lo, hi, best = 1.0, upper_bound_gib, 0.0
    while hi - lo >= RESOLUTION_GIB:
        mid = (lo + hi) / 2
        if allocation_survives_write(mid):
            best, lo = mid, mid
        else:
            hi = mid
    return best, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-gib", type=float, default=DEFAULT_MAX_GIB)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible -- activate the lab venv (scripts/activate-lab.ps1)")

    props = torch.cuda.get_device_properties(0)
    reported_gib = props.total_memory / 1024**3
    free_b, _ = torch.cuda.mem_get_info()

    print(f"device: {torch.cuda.get_device_name(0)}  arch: {getattr(props, 'gcnArchName', '?')}")
    print(f"reported_pool_GiB: {reported_gib:.2f}")
    print(f"mem_get_info_free_GiB: {free_b / 1024**3:.2f}")

    start = time.perf_counter()
    ceiling, saturated = max_writable_allocation_gib(args.max_gib)
    search_s = time.perf_counter() - start

    if saturated:
        print(
            f"max_writable_allocation_GiB: >={ceiling:.0f}  "
            f"** SEARCH SATURATED at --max-gib -- this is the bound, not the ceiling **"
        )
    else:
        print(f"max_writable_allocation_GiB: {ceiling:.1f}")
    print(f"search_seconds: {search_s:.1f}")

    print(
        json.dumps(
            {
                "reported_pool_gib": round(reported_gib, 2),
                "max_writable_allocation_gib": round(ceiling, 1),
                "search_saturated": saturated,
                "search_upper_bound_gib": args.max_gib,
            }
        )
    )


if __name__ == "__main__":
    main()
