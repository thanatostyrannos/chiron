# docs/ — rig design

**Authored in the Rig Design phase.** The rig is a controlled-comparison instrument:
its job is to make "does change X help?" answerable with evidence, cheaply and
repeatably. Planned documents (named for content, not numbered):

- `experiment-unit-economics.md` — the **G3 gate for the rig**. Unit = one ablation
  run. Cost it (GPU-hours × verified $/hr, storage, wall-clock) at 1 / 20 / 200 runs
  and name what breaks between those scales. Label every input measured / benchmarked
  / assumed; a ±30% flip promotes an assumption to a measurement priority.
- `evidence-standard.md` — what an ablation must produce to count (fixed seeds,
  matched token budgets & param counts, CIs over ≥3 seeds, pre-registered G2 card).
- `system-architecture.md` — the four systems and their boundaries; experiment
  lifecycle; config→run→artifact→report flow. Mnemosyne separable from Proteus.
- `proteus-config-space.md` — the configurable surface (per-layer attention type,
  interleaving ratio, MoE gating, positional scheme). Every axis exists to be ablated.
- `mnemosyne-cache-interface.md` — the memory subsystem's public contract: how
  Proteus requests/releases cache, what policies plug in, what telemetry every policy
  must emit for attribution, and the invariants a policy must not break.
- `corpus-and-probes.md` — small permissive corpus + recall/retrieval probes.
- `argus-telemetry-schema.md` — JSONL-first metrics schema and post-hoc diagnosis.
- `research-roadmap.md` — Mermaid gantt feeding `ABLATION_LOOP_PROMPT.md`.

Subdirectories:
- `adr/` — Architecture Decision Records. **Immutable once Accepted.** See
  `adr/README.md` for the register and the hash-enforcement rule.
- `diagrams/` — Mermaid sources (`*.mmd`).
