# Environment Report

Generated: 2026-07-26 11:58  |  **18 / 19 checks passed**

Platform target: native Windows, AMD Strix Halo (gfx1151). Not WSL2 — the ROCm
pool under WSL2 is clamped to the .wslconfig memory value and cannot reach
dedicated VRAM (ROCm issue #6022).

| Check | Value | Expected | OK |
|---|---|---|:--:|
| OS | `Microsoft Windows 11 Pro 10.0.26200` | Windows 11 | YES |
| CPU | `AMD RYZEN AI MAX+ 395 w/ Radeon 8060S          ` | Ryzen AI MAX+ 395 | YES |
| GPU | `AMD Radeon(TM) 8060S Graphics` | Radeon 8060S | YES |
| GPU driver | `32.0.23033.5002` | Adrenalin 26.x | YES |
| System RAM (GB) | `32` | ~128 minus UMA reservation | YES |
| git | `git version 2.54.0.windows.1` | any | YES |
| gh CLI | `gh version 2.92.0 (2026-04-28)` | any | YES |
| gh auth | `authenticated` | authenticated | YES |
| Python | `Python 3.12.10` | 3.11 or 3.12 | YES |
| uv | `uv 0.11.19 (7b2cff1c3 2026-06-03 x86_64-pc-windows-msvc)` | any | YES |
| Repo path has no spaces | `clean: C:\projects\School\chiron` | no spaces | YES |
| Python env path has no spaces | `clean: C:\venvs\lab` | no spaces | YES |
| torch importable | `2.12.0a0+rocm7.13.0a20260313` | 2.9+ with rocm tag | YES |
| torch sees GPU | `True` | True | YES |
| device name | `AMD Radeon(TM) 8060S Graphics` | AMD Radeon 8060S | YES |
| arch is gfx1151 | `gfx1151` | gfx1151 | YES |
| hipBLASLt configured | `set: C:\venvs\lab\Lib\site-packages\_rocm_sdk_libraries_gfx1151\bin\hipblaslt\library` | set | YES |
| Max GPU allocation | `95 GiB alloc-only (untouched; see measure_capacity_ceiling.py)` | close to BIOS UMA FB Size (target 96GB) | YES |
| huggingface-cli | `ERROR: The term 'huggingface-cli' is not recognized as the name of a cmdlet, function, script file, or operable program. Check the spelling of the name, or if a path was included, verify that the path is correct and try again.` | any | **NO** |

## Remediation required

- **huggingface-cli** — pip install huggingface_hub[cli] — needed only for gated model downloads. Then: huggingface-cli login

