# configs/ — experiment configuration (YAML)

Config is YAML, deserialized into one typed config object. **Every ablation axis is a
config field** — the config surface IS the experimental surface. Configs are named
for what they configure, never numbered (`swa-4to1-ctx32k.yaml`, not `exp3.yaml`).

Conventions:
- A **CPU-fallback config is required for every training config** (the Z13 GPU stack
  is unproven until the Hardware Validation Gate; scaffold/design work must run without it).
- Determinism first: every run is seeded; dataloaders are resumable.
- Matched param counts and token budgets across arms of a comparison, or the arms are
  not comparable.

Populated from the Rig Design phase onward.
