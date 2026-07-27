"""Does bf16 attention-weighted-sum accuracy degrade with sequence length on gfx1151?

A long-context attention output is a convex combination of N value vectors. In bf16 the
accumulation order and accumulator width decide how much of a small weight survives. If
the GPU's reduction is materially worse than the CPU's at large N, then every
long-context recall result on this machine is confounded by arithmetic before any
memory policy is applied -- which is `bf16-numerics-unproven` biting exactly where the
lab's research question lives.

Method: draw V once in float64 on CPU, share the SAME bits with both devices, form a
softmax weight vector with one deliberately dominant entry, cast to bf16, and compare
each device's p@V against the float64 reference.

Reports the GPU/CPU error ratio. ~1.0 means the devices agree; materially above 1.0 is a
finding, not a rounding detail.

Usage:
    python scripts/measure_bf16_reduction_error.py
"""

from __future__ import annotations

import json
import math

import torch

HEAD_DIM = 128
SEEDS = (1337, 1338, 1339)
LENGTHS = (8192, 131072, 1048576)
NEEDLE_SHARE = 0.5  # target attention mass on index 0


def relative_error(seq_len: int, seed: int, device: str) -> float:
    """||(p@V)_bf16 - (p@V)_fp64|| / ||(p@V)_fp64||, with p,V bit-identical across devices."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values_fp64 = torch.randn(seq_len, HEAD_DIM, dtype=torch.float64, generator=generator)

    # One logit raised so index 0 carries NEEDLE_SHARE of the mass; the rest are 0.
    delta = math.log((seq_len - 1) * NEEDLE_SHARE / (1.0 - NEEDLE_SHARE))
    logits = torch.zeros(seq_len, dtype=torch.float64)
    logits[0] = delta
    weights_fp64 = torch.softmax(logits, dim=0)

    reference = weights_fp64 @ values_fp64  # fp64 on CPU, the ground truth

    weights = weights_fp64.to(torch.bfloat16).to(device)
    values = values_fp64.to(torch.bfloat16).to(device)
    got = (weights @ values).to(torch.float64).cpu()

    return (torch.linalg.vector_norm(got - reference) / torch.linalg.vector_norm(reference)).item()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible -- activate the lab venv")

    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    print(f"needle share {NEEDLE_SHARE}, head_dim {HEAD_DIM}, seeds {SEEDS}\n")
    print(f"{'N':>9} {'seed':>6} {'cpu_rel_err':>12} {'gpu_rel_err':>12} {'gpu/cpu':>8}")

    rows = []
    for seq_len in LENGTHS:
        for seed in SEEDS:
            cpu_err = relative_error(seq_len, seed, "cpu")
            gpu_err = relative_error(seq_len, seed, "cuda")
            ratio = gpu_err / cpu_err if cpu_err else float("nan")
            rows.append(
                {"n": seq_len, "seed": seed, "cpu": cpu_err, "gpu": gpu_err, "ratio": ratio}
            )
            print(f"{seq_len:>9} {seed:>6} {cpu_err:>12.3e} {gpu_err:>12.3e} {ratio:>8.2f}")
            torch.cuda.empty_cache()

    print()
    for seq_len in LENGTHS:
        ratios = [r["ratio"] for r in rows if r["n"] == seq_len]
        mean = sum(ratios) / len(ratios)
        verdict = "AGREES" if mean < 1.5 else "GPU MATERIALLY WORSE"
        print(f"N={seq_len:>9}  mean gpu/cpu ratio {mean:5.2f}   {verdict}")

    print(json.dumps({"rows": rows}))


if __name__ == "__main__":
    main()
