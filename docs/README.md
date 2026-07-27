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

## Written so far

- **`model-architecture.md`** — the architecture and system-layout diagrams, 11 of them,
  each grounded in a config key, a `file:line`, or an `ASSUMPTIONS.md` row (336 grounding
  references total). Start here for a picture of how any of this fits together.
- **`model-architecture.html`** — the same document as a self-contained page with every
  diagram inlined as SVG. No network, no JS; open it from disk. Generated — edit the
  markdown and run `uv run --script scripts/build_architecture_html.py`.
- **`diagrams/`** — the Mermaid sources, render-validated by
  `scripts/validate_diagrams.py`.
- **`adr/`** — 4 architecture decision records, 3 accepted and hash-frozen.

Still to be written in the Rig Design phase: `experiment-unit-economics.md`,
`evidence-standard.md`, `system-architecture.md`, `proteus-config-space.md`,
`mnemosyne-cache-interface.md`, `corpus-and-probes.md`, `argus-telemetry-schema.md`,
`research-roadmap.md`. Several now have a diagram waiting for the prose.
