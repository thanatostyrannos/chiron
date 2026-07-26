# Environment Report

Generated: 2026-07-24 09:40  |  **13 / 19 checks passed**

Platform target: native Windows, AMD Strix Halo (gfx1151). Not WSL2 — the ROCm
pool under WSL2 is clamped to the .wslconfig memory value and cannot reach
dedicated VRAM (ROCm issue #6022).

| Check | Value | Expected | OK |
|---|---|---|:--:|
| OS | `Microsoft Windows 11 Pro 10.0.26200` | Windows 11 | YES |
| CPU | `AMD RYZEN AI MAX+ 395 w/ Radeon 8060S          ` | Ryzen AI MAX+ 395 | YES |
| GPU | `AMD Radeon(TM) 8060S Graphics` | Radeon 8060S | YES |
| GPU driver | `32.0.23033.5002` | Adrenalin 26.x | YES |
| System RAM (GB) | `112` | ~128 minus UMA reservation | YES |
| git | `git version 2.54.0.windows.1` | any | YES |
| gh CLI | `gh version 2.92.0 (2026-04-28)` | any | YES |
| gh auth | `authenticated` | authenticated | YES |
| Python | `Python 3.12.10` | 3.11 or 3.12 | YES |
| uv | `uv 0.11.19 (7b2cff1c3 2026-06-03 x86_64-pc-windows-msvc)` | any | YES |
| Repo path has no spaces | `clean: C:\projects\School\chiron` | no spaces | YES |
| Python env path has no spaces | `clean: C:\Users\solar\AppData\Local\Programs\Python\Python312` | no spaces | YES |
| torch importable | `2.11.0+cu128` | 2.9+ with rocm tag | YES |
| torch sees GPU | `MISSING - no device` | True | **NO** |
| device name | `MISSING` | AMD Radeon 8060S | **NO** |
| arch is gfx1151 | `` | gfx1151 | **NO** |
| hipBLASLt configured | `MISSING - unset (GEMM may run ~5x slow)` | set | **NO** |
| Max GPU allocation | `MISSING - no GPU` | close to BIOS UMA FB Size (target 96GB) | **NO** |
| huggingface-cli | `ERROR: The term 'huggingface-cli' is not recognized as the name of a cmdlet, function, script file, or operable program. Check the spelling of the name, or if a path was included, verify that the path is correct and try again.` | any | **NO** |

## Remediation required

- **torch sees GPU** — Install AMD's gfx1151 nightlies into a clean venv:
    python -m venv c:\venvs\lab && c:\venvs\lab\Scripts\activate
    pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ "rocm[libraries,devel]"
    pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchvision torchaudio

- **device name** — Install AMD's gfx1151 nightlies into a clean venv:
    python -m venv c:\venvs\lab && c:\venvs\lab\Scripts\activate
    pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ "rocm[libraries,devel]"
    pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchvision torchaudio

- **arch is gfx1151** — If arch is wrong or unknown, you have a generic build. Reinstall from the gfx1151 index above.

- **hipBLASLt configured** — Locate the hipblaslt library dir in your ROCm install and set HIPBLASLT_TENSILE_LIBPATH. Also set TORCH_BLAS_PREFER_HIPBLASLT=1. A misconfigured path drops GEMM from ~33 to ~6 TFLOPS.

- **Max GPU allocation** — If far below the BIOS UMA setting: raise UMA FB Size in BIOS, confirm the driver exposes it, and re-run. This number is the ceiling for every long-context experiment — record it.

- **huggingface-cli** — pip install huggingface_hub[cli] — needed only for gated model downloads. Then: huggingface-cli login

