# We will build an attribution instrument rather than propose another eviction policy

Status:   Proposed
Date:     2026-07-26
Deciders: Founder (owner — pending review), Claude (research staff)

## Context

`research/synthesis.md` was built from two independent drafts — a decision lens and a
contribution lens, run blind to each other — plus an adversarial completeness critic.
Both lenses converged on the same recommendation and named the same riskiest assumption
without seeing one another. That convergence is the strongest evidence in the document.

Three findings drive it.

**The field's binding constraint is measurement, not policy.** `open-problems-ranked.md`
scores attribution P5·T5·E5, the only 5/5/5 on the list. There are ~30 published eviction
policies and no dominance result. Four documented cases exist where the outcome metric
held while the mechanism broke: refusals down 15.2% at 1.03x perplexity `[C]`
(2606.09864); specific instructions dropped entirely under StreamingLLM/SnapKV/TOVA/H2O
while LongBench looked fine `[C]` (2510.00231); single-turn rankings not surviving
multi-turn cache reuse `[C]` (2412.10319); mean-aggregated rankings not surviving
worst-case aggregation `[C]` (2510.13334). Several groups argue most of PyramidKV's
reported gain comes from SnapKV's observation window rather than the per-layer budget it
claims credit for.

**Attribution needs a full-cache oracle on every probe** — you must run the expensive
thing you were trying to avoid. At 300M that is ~600 MB of weights against a `[M]`
≥62 GiB fast tier. At 70B it is unaffordable. Small scale is the enabling condition
here, not a compromise.

**The standard eval is adversely selected against the mechanism.** A
needle-in-a-haystack needle is a low-frequency, high-salience span that attracts
attention mass — exactly what heavy-hitter eviction retains. A policy can shed most of
the cache, destroy ordinary long-range dependence, and still pass. A twelve-month search
of the eviction literature found no needle-removed control.

The alternative considered and rejected: implement `mnemosyne-h2o` and friends, and
compete on policy quality. Rejected because it adds a thirty-first policy to a field
with no way to tell which of the thirty works, which is negative-value work.

## Decision

We will make the **attribution instrument the lab's deliverable**, and add no new
eviction policy until we can measure one honestly.

Concretely, in order: (1) close and widen the Hardware Validation Gate; (2) build
Mnemosyne's full-cache oracle-diff harness with a seed-to-seed null distribution;
(3) calibrate every eval by fault injection before it is permitted to certify an arm.
Then run the tier-ratio experiment the BIOS carve-out makes uniquely possible here.

Mnemosyne's public interface will expose **three plug points** — write-time admission,
deferred eviction, read-time selection — because a single `score(...) -> subset` hook
cannot host the existing literature: FastKV alters the forward pass, RocketKV's second
stage needs the current query, and KVpop needs a staging buffer.

We park, with un-park triggers written down: distributed/disaggregated KV and CXL
pooling (no collectives), hybrid-ratio search as a training programme, sub-4-bit KV
quantization (confounded by `bf16-numerics-unproven`), agent-memory security, and RLVR
as a capability programme.

## Consequences

**Makes easy.** A defensible contribution that does not require competing on compute:
the instrument is affordable precisely because our models are small. It also gives the
lab a publishable methodology result — fault-injection calibration of memory evals —
before any research arm exists.

**Makes hard.** It defers anything that looks like a headline architecture result. The
first deliverable is a harness, which is less satisfying to build and harder to explain
than "our policy beats H2O".

**Forecloses.** Nothing technically, but it sets the lab's identity: if attribution
turns out to be uninteresting, the pivot cost is the harness.

**Riskiest assumption, stated because it can kill this.** That distributional divergence
from a full-cache oracle measures anything decision-relevant. Divergence and task
accuracy can dissociate in both directions — a policy can shift the output distribution
without flipping any argmax, or flip one critical token at negligible average KL. The
cheapest decisive test comes before the harness: drop a *known* cache entry and check
whether per-token KL localises to it and moves only when the recoverable token moves.
Run it in fp32 to avoid the bf16 confound; it depends on determinism, which is a gate
item. If it fails, this ADR is superseded rather than patched.

**Status is `Proposed`**, not Accepted, because `research/synthesis.md` carries the same
status and awaits founder review. It is written now so the reasoning is frozen at the
moment the choice was made rather than reconstructed later.
