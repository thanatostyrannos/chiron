"""Does scaled_dot_product_attention actually avoid materialising the score matrix?

`torch.backends.cuda.flash_sdp_enabled()` returning True says a backend is *permitted*,
not that it was *used*. If SDPA silently falls back to the math backend, attention
allocates a B x nh x T x T score tensor, activation memory grows quadratically in
sequence length, and long-context work walks into the measured >=32 GiB single-tensor
fault -- which on this machine presents as a silent hang at 0 CPU, not an OOM.

This measures the thing rather than asking the API: it fits saved-activation bytes
against T and reports the quadratic coefficient. A coefficient near zero means the score
matrix was never kept; a coefficient matching B*nh*T^2*dtype_bytes means it was.

Usage:
    python scripts/measure_attention_memory_path.py
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python scripts/measure_attention_memory_path.py
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F

BATCH = 4
HEADS = 8
HEAD_DIM = 64
SEQ_LENS = (128, 256, 512)
DTYPE = torch.bfloat16


def saved_activation_bytes(seq_len: int) -> int:
    """Total bytes autograd retains for backward during one SDPA forward."""
    seen: dict[int, int] = {}

    def pack(t: torch.Tensor) -> torch.Tensor:
        # Dedup by storage pointer; parameters are excluded by construction here
        # because q/k/v are the only leaves and we subtract them below.
        seen[t.data_ptr()] = t.numel() * t.element_size()
        return t

    shape = (BATCH, HEADS, seq_len, HEAD_DIM)
    q = torch.randn(shape, device="cuda", dtype=DTYPE, requires_grad=True)
    k = torch.randn(shape, device="cuda", dtype=DTYPE, requires_grad=True)
    v = torch.randn(shape, device="cuda", dtype=DTYPE, requires_grad=True)
    leaf_bytes = 3 * q.numel() * q.element_size()

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out.sum().backward()

    total = sum(seen.values())
    del q, k, v, out
    torch.cuda.empty_cache()
    return max(0, total - leaf_bytes)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible -- activate the lab venv")

    env = os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "<unset>")
    print(f"torch {torch.__version__}")
    print(f"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = {env}")
    for name in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled"):
        fn = getattr(torch.backends.cuda, name, None)
        print(f"  torch.backends.cuda.{name}() = {fn() if fn else 'n/a'}")

    print(f"\n{'T':>6} {'saved_MiB':>11} {'bytes/T^2':>11}")
    measured = {}
    for seq_len in SEQ_LENS:
        saved = saved_activation_bytes(seq_len)
        measured[seq_len] = saved
        print(f"{seq_len:>6} {saved / 1024**2:>11.1f} {saved / seq_len**2:>11.1f}")

    # Difference-of-differences isolates the T^2 term without fitting machinery.
    t1, t2 = SEQ_LENS[0], SEQ_LENS[-1]
    quad = (measured[t2] - measured[t1]) / (t2**2 - t1**2)
    predicted = BATCH * HEADS * torch.tensor([], dtype=DTYPE).element_size()

    print(f"\nquadratic coefficient : {quad:.1f} bytes/T^2")
    print(f"score-matrix predicts : {predicted:.1f} bytes/T^2  (B*nh*dtype_bytes)")
    materialised = quad > 0.5 * predicted
    print(f"score matrix retained : {materialised}")
    if materialised:
        print(
            "\nATTENTION IS QUADRATIC IN MEMORY HERE. The B*nh*T^2 score tensor reaches the\n"
            "measured 32 GiB single-tensor fault -- which hangs silently at 0 CPU."
        )
    print(json.dumps({
        "env": env, "quadratic_bytes_per_t2": round(quad, 1),
        "predicted_if_materialised": float(predicted),
        "score_matrix_retained": bool(materialised),
        "saved_bytes": measured,
    }))


if __name__ == "__main__":
    main()
