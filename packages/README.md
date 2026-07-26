# packages/ — the uv workspace members

Three separately-distributable packages. Each has its own `pyproject.toml`, version,
and `tests/`. One shared venv via `uv sync` from the repo root.

| Package | System | Depends on | Never imports |
|---|---|---|---|
| `mnemosyne/` | memory subsystem — **the research contribution** | torch | proteus, themis |
| `proteus/` | model architecture | torch, mnemosyne | themis |
| `themis/` | ablation rig (contains `argus/` telemetry, `data/` loaders) | torch, proteus, mnemosyne | — |

## The boundary rule (mechanical, not aspirational)

```
mnemosyne  →  torch
proteus    →  torch, mnemosyne
themis     →  torch, proteus, mnemosyne
```

Mnemosyne must stay separable from Proteus: if memory management only works against
our specific model it is an implementation detail; if it can be pointed at a
different model it is a contribution. Because `mnemosyne/pyproject.toml` does not
declare proteus, an accidental import fails at resolution — and
`tests/test_package_boundaries.py` fails the build on any `import proteus` /
`import themis` in mnemosyne source. Do **not** add the dependency "temporarily."

**Separability acceptance test** (run at the `mnemosyne-core` milestone, not every
commit): build the mnemosyne wheel, install it into a clean venv containing only
torch, run its test suite green. If that fails, the boundary has leaked.

Cross-package integration tests live in the repo-root `tests/`, not here.
