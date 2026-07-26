# BLOCKERS — what is stopping work right now

A **register**: rows are appended and their status updated; rows are never deleted.
Resolved blockers stay, marked `resolved`, with the date and what unblocked them.
Anything here that is really an untested belief belongs in `ASSUMPTIONS.md` instead —
this file is for things that stop work, not things that might be wrong.

| Blocker | Blocks | Owner | Status | Date |
|---|---|---|---|---|
| gfx1151 ROCm torch not installed — the venv still has no AMD-capable torch (`ENVIRONMENT.md`: 6/19 checks red, all this stack). | Hardware Validation Gate, every training or capacity run. **Does not block** Reference Library → Ablation Backlog, which are CPU-only. | Claude (install is pre-authorized) | **resolved** — `torch 2.12.0a0+rocm7.13.0a20260313` live, preflight 18/19 | 2026-07-26 |
| `hardware-capacity-ceiling` unmeasured — the largest allocation that actually reaches the UMA pool is unknown (`ASSUMPTIONS.md`). | Every long-context / KV-capacity experiment; the sizing of the whole ablation backlog. | Claude, after the install above | **superseded** — the binding number turned out to be the fast-tier size (≥62 GiB), not the allocation ceiling. Exact writable ceiling still unmeasured but no longer on the critical path. | 2026-07-26 |
| BIOS UMA FB Size unconfirmed — target 96 GB. Cannot be set or read from software. | The capacity ceiling above; a low setting caps it regardless of software. | Owner (BIOS, requires reboot) | **resolved** — set to 96 GB; doubled the fast tier (30 → ≥62 GiB) | 2026-07-26 |
| Single tensors ≥32 GiB hang (0 CPU, no error) or raise `hipErrorLaunchFailure`. | Any long-context / KV-cache experiment allocating a single buffer ≥32 GiB. Workaround: keep individual buffers under 32 GiB. | Claude — Hardware Validation Gate, needs its own pre-registration | **open** | 2026-07-26 |
| GitHub remote not created — `gh repo create` needs a public/private decision, and a public push is on CLAUDE.md's ask-first list. `gh auth` is already good. | Off-machine backup and any published artifact. Blocks nothing local. | Owner | **open** | 2026-07-26 |
