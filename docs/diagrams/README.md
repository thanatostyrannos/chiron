# docs/diagrams/ — Mermaid sources

One `.mmd` per diagram, embedded into `docs/model-architecture.md`. Mermaid rather than
Visio or Lucidchart on purpose: a `.vsdx` is an opaque binary that does not diff, cannot
be reviewed in a pull request, and does not render on GitHub. It would be the only
unverifiable artifact in a repo whose whole discipline is checkable text.

Every diagram is **grounded** — each element traces to a config key, a `file:line` in
`research/reference/`, or a row in `ASSUMPTIONS.md`. A diagram drawn from memory is
worse than no diagram: it looks equally authoritative and nothing checks it.

## Validate before committing

```
python scripts/validate_diagrams.py                                # render-check all
python scripts/validate_diagrams.py --svg docs/diagrams/rendered   # export SVG
```

Validation is by **actual render** via mermaid-cli, not a syntax heuristic. A diagram
that does not parse still looks like a diagram in source — it fails as a grey error box
in front of a reader. Three such blocks were already committed in `v0.7.0` before this
check existed.

Needs node. The script resolves `mmdc` from PATH, then a local `node_modules`, then
`npx -y @mermaid-js/mermaid-cli`. Set `CHIRON_MMDC_ROOT` to point at an out-of-repo
install.

## Gotchas that have actually bitten

- **A semicolon ends a Mermaid statement.** `A->>B: clip_grad_norm_(1.0); record norms`
  parses the text after `;` as a new statement and fails. Use a comma. This was the cause
  of all three broken blocks in `v0.7.0`. Parentheses, `<`, `->` inside message text and
  `<br/>` in notes are all fine — the semicolon was the only real offender.
- **Line numbers in labels are pinned** to the revisions in `research/reference/PROVENANCE.md`.
  Re-fetching upstream moves them. `scripts/verify_code_pointers.py` catches it.

## The diagrams

| File | Shows |
|---|---|
| `laguna-decoder-stack.mmd` | The 48-layer GSSS interleave, 12 global + 36 sliding at window 512 |
| `laguna-decoder-block.mmd` | One block: norms, QK-norm, per-layer-type RoPE, per-head output gating, MoE slot |
| `laguna-moe-routing.mmd` | Sigmoid routing, aux-loss-free bias correction, top-10 of 256, shared expert |
| `attention-variants-and-kv-cost.mmd` | MHA / GQA / MQA / MLA as four values of one factor, with per-token bytes |
| `proteus-config-surface.mmd` | Our ablation axes — the config surface *is* the experimental surface |
| `four-systems-and-boundaries.mmd` | Chiron / Proteus / Mnemosyne / Themis and the enforced dependency direction |
| `experiment-lifecycle.mmd` | One ablation run, with pre-registration as a gate *before* the run |
| `mnemosyne-cache-interface.mmd` | The three plug points and the telemetry every policy must emit |
| `memory-hierarchy-measured.mmd` | Our measured tiers against the datacenter assumption the literature uses |
| `paged-attention-block-table.mmd` | Logical-to-physical blocks, prefix sharing, and where the VM analogy breaks |
| `attribution-oracle-diff.mmd` | The oracle-diff harness and its fault-injection calibration loop |
