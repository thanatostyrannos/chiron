# The open-weights landscape, mid-2026

This note settles three things. **First**, that "open" in 2026 is a five-rung ladder and
essentially the entire frontier — DeepSeek V4, GLM-5.2, Kimi K3, Qwen3.5/3.6, Inkling,
Gemma 4, Mistral 3, gpt-oss, and our own reference model Laguna — sits on **rung 2:
downloadable weights under a standard permissive licence, with no training data, no
training code, no intermediate checkpoints and no logs**, so the only questions an
outsider can answer from them are questions about a *frozen forward pass*. **Second**,
that OpenMDW-1.1 (Laguna) is genuinely more permissive than Apache-2.0 in the two places
that matter to this lab — it explicitly disclaims any restriction on model **outputs**,
and its grant reaches **database and trade-secret rights** as well as copyright and
patent — but pays for it with a *broader* litigation-termination clause and weaker
notice mechanics, and it compels no data release whatsoever. **Third**, that the labs
publishing enough for an outsider to genuinely build on are a short, non-frontier list —
Ai2 (OLMo 3), Swiss AI (Apertus), AMD (Instella), LLM360, EleutherAI's Common Pile /
Comma — and that this lab's experimental substrate must come from *that* list even
though its architectural reference comes from rung 2.

---

## 1. The ladder, and where the software analogy breaks

The instinct from thirty years of systems work is to map this onto binary-vs-source. It
half-works. Use the LF AI & Data **Model Openness Framework** as the formal scaffold —
17 lifecycle components, three classes: Class III *Open Model* (weights + basic docs),
Class II *Open Tooling* (+ training/eval code and key datasets), Class I *Open Science*
(everything) `[C]` arXiv `2403.13784` (Mar 2024, v6). In practice the market has
subdivided Class III, so the operational ladder is:

| Rung | What you get | What you can actually answer |
|---|---|---|
| **0. API-only** | tokens | Nothing about mechanism. Ceiling reference only. |
| **1. Weights, bespoke licence** | weights + card, licence with conditions | Inference behaviour — subject to use restrictions that can bite research. |
| **2. Weights, permissive licence** | weights + config + often a runnable `modeling_*.py` | The **exact forward pass**. Architecture, cache geometry, routing, positional scheme. |
| **3. + recipe** | technical report with data-mix proportions, optimizer, ablations | The *hypothesis space* of training decisions. You cannot test any of it. |
| **4. Full-stack reproducible** | + corpus, training code, intermediate checkpoints, logs, eval harness | Counterfactuals. Training **dynamics**. Contamination audits. |

**Where the analogy holds.** Rung 2 really is "binary plus header files." Rung 4 really
is "source plus build system." And the reason rung 4 is rare is the same reason nobody
ships a reproducible build of a proprietary firmware image: the inputs are the asset.

**Where it breaks, three times, and each break is load-bearing.**

*Break 1 — there is no bit-exact rebuild.* A reproducible software build is a hash
match. Even a complete rung-4 release does not let you regenerate the weights
bit-identically: reduction order, kernel selection, and cluster topology all move the
result. Poolside's own report describes hunting non-determinism in Muon's Newton–Schulz
iterations and finding a checkpointer silently inserting an extra cast into the optimizer
state-dict path `[C]` (Laguna M.1/XS.2 Technical Report, 25 May 2026, §3.1). So
"reproducible" for a model means *statistically* reproducible — same recipe, same loss
curve within noise — which is a strictly weaker contract than a build hash and is why
rung-4 releases publish **intermediate checkpoints**: the checkpoint sequence is the
closest thing a model has to a build log.

*Break 2 — the compile step costs millions.* Source without a compiler is not much use.
Laguna S 2.1 was trained on 4,096 H200s in under nine weeks `[C]`
([VentureBeat, 21 Jul 2026](https://venturebeat.com/infrastructure/poolside-drops-laguna-s-2-1-an-open-weight-coding-model-that-beats-rivals-10x-its-size));
M.1 and XS.2 on 6,144 and 2,048 H200s respectively `[C]` (technical report §3.1). For a
one-GPU lab, the **checkpoints** are worth more than the data, because they are the only
artifact that exposes training dynamics without paying for training. This inverts the
usual open-source priority ordering and it is the single most useful thing in this note.

*Break 3 — the licence covers the artifact, not the inputs.* An SBOM makes upstream
licence provenance auditable for software. No frontier open-weight model discloses its
corpus, so no equivalent audit exists, and the model licences say so out loud. OpenMDW-1.1
puts it in capitals: **"YOU ARE SOLELY RESPONSIBLE FOR (1) CLEARING RIGHTS OF OTHER
PERSONS THAT MAY APPLY TO THE MODEL MATERIALS"** `[M]` (local `LICENSE.md`,
`models/laguna-s` @ `b0a9fd7c850e`, PROVENANCE). The EU AI Act's Article 53(1)(d) public
training-content summary is the first regulatory attempt to force a partial model SBOM;
the Commission template landed 24 Jul 2025 and the AI Office gains full enforcement
powers 2 Aug 2026 `[C]`
([EC guidelines for GPAI providers](https://digital-strategy.ec.europa.eu/en/policies/guidelines-gpai-providers)).

---

## 2. Who ships what, mid-2026

Param counts marked `[C]` come from vendor cards or press and are secondary unless we
hold the artifact locally.

| Provider | Flagship open release | Licence | Rung |
|---|---|---|---|
| **poolside** | Laguna S 2.1 118B-A8B (Jul 2026); XS 2.1 33B-A3B | **OpenMDW-1.1** | 3 (report, no data/code) |
| DeepSeek | V4-Pro 1.6T-A49B, V4-Flash 284B-A13B (24 Apr 2026) | MIT | 3 (arXiv `2606.19348`) |
| Z.ai | GLM-5.2 (~744–753B-A40B, weights 16 Jun 2026) | MIT | 2 |
| Moonshot | Kimi K3 ~2.8T (16 Jul 2026; weights 27 Jul) | Modified MIT | 2 |
| Alibaba | Qwen3.5-397B-A17B (16 Feb 2026), Qwen3.6-35B-A3B | Apache-2.0 | 2 |
| | Qwen3.7-Max (May 2026) | **closed, API-only** | 0 |
| Thinking Machines | Inkling 975B-A41B + Small 276B-A12B (15 Jul 2026) | Apache-2.0 | 2 |
| MiniMax | M2.5 (Feb 2026) | MiniMax Model License (mod-MIT + prohibited uses) | 2/1 |
| NVIDIA | Nemotron 3 Nano/Super/Ultra (Dec 2025 – Jun 2026) | NVIDIA Open Model License → **OpenMDW-1.1** | 3+ (datasets + recipes) |
| Meta | Llama 4 Scout / Maverick | Llama 4 Community License | 1 |
| Google | Gemma 4 (2 Apr 2026) | **Apache-2.0** (was Gemma Terms of Use) | 2 |
| Mistral | Mistral 3 family | Apache-2.0 | 2 |
| OpenAI | gpt-oss-120b / 20b | Apache-2.0 | 2 (card = arXiv `2508.10925`) |
| **Ai2** | **OLMo 3** 7B/32B (Nov 2025), 3.1 (Dec 2025) | Apache-2.0 + Dolma 3 + OLMo-core | **4** |
| **Swiss AI** | **Apertus** 8B/70B | full stack, 15T tokens, 1800+ languages | **4** |
| **AMD** | **Instella** 3B (+ Long, Math) | weights + data + code, trained on MI300X | **4** |
| **LLM360** | K2-65B "DIAMOND" | full stack | **4** |
| **EleutherAI et al.** | Comma v0.1-1T / 2T on Common Pile | full stack, **openly-licensed corpus** | **4** |

`[C]` for the release rows: [Epoch AI](https://epoch.ai/data-insights/open-closed-eci-gap),
vendor cards, and dated press linked in Sources. `[M]` for Laguna: config and licence read
from the pinned clones at `b0a9fd7c850e` / `205dc65dd4bd`.

Two structural facts about this table.

**The frontier of *openness* and the frontier of *capability* are different lists.** Every
rung-4 entry is below the capability frontier. Epoch AI measures the open-vs-closed gap at
**4 months / 8 ECI points (90% CI 7–11)** over 1 Jan – 28 May 2026, *widened* from the
3 months they measured through Oct 2025, and note two biases that make their estimate an
under-statement `[C]` (Epoch, 29 May 2026). The popular narrative that the gap is closing
is not what their number says.

**Direction of travel on licences is toward the boring permissive ones.** Gemma abandoned
its bespoke Terms of Use with a Prohibited Use Policy for plain Apache-2.0 at Gemma 4
`[C]` ([Google, 2 Apr 2026](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/));
NVIDIA announced it will move Cosmos, Isaac GR00T, Ising and Nemotron to OpenMDW-1.1
`[C]` ([Linux Foundation, 28 May 2026](https://www.linuxfoundation.org/press/linux-foundation-releases-openmdw-1.1-nvidia-adopts-openmdw-for-cosmos-isaac-gr00t-ising-and-nemotron-ai-model-families)).
Note poolside went the other way in a sense: Laguna **XS.2 shipped under Apache-2.0**
`[C]` (technical report §1), while the 2.1 generation ships under OpenMDW-1.1 `[M]`.

---

## 3. Licence reality: what we may actually do

### OpenMDW-1.1 (Laguna), read from the local text

The whole licence is 49 lines and we hold it at a pinned revision. `[M]` all quotes below
from `research/reference/models/laguna-s/LICENSE.md` @ `b0a9fd7c850e`.

- **Scope is elastic.** "Model Materials" = the model (architecture and parameters) plus
  "all related artifacts (including associated data, documentation and software) **that
  are provided to you hereunder**." The licence governs what the provider chose to ship;
  OpenMDW's own FAQ is explicit: *"OpenMDW does not mandate which components you release"*
  `[C]` ([openmdw.ai](https://openmdw.ai/)). **A model can be OpenMDW-licensed and rung 2.
  Laguna is.**
- **The grant is broader than Apache-2.0's.** "permission … to deal in the Model Materials
  without restriction, including under all **copyright, patent, database, and trade
  secret** rights included or embodied therein." Apache-2.0 grants copyright and patent
  only. The EU *sui generis* database right is not a copyright, and Apache-2.0 arguably
  does not license it — which is precisely why a model-specific licence exists at all.
- **Conditions are minimal.** On redistribution: retain a copy of the agreement and
  retain copyright/origin notices. That is the complete list. Apache-2.0 §4(b) additionally
  requires **prominent change notices on modified files** and §4(d) requires NOTICE-file
  propagation. OpenMDW has neither. Lower compliance burden; also less provenance.
- **Retaliation is broader, not narrower.** Filing or "voluntarily participat[ing] in" a
  suit asserting the Model Materials infringe **"any patent or copyright"** terminates
  **all** rights and grants. Apache-2.0 §3 terminates only the *patent* licence and only
  for patent claims about the Work. This patent→"patent or copyright" widening **is** the
  1.0→1.1 change `[C]` ([OpenMDW/OpenMDW](https://github.com/OpenMDW/openmdw)).
- **Outputs are explicitly unencumbered.** "This agreement does not impose any restrictions
  or obligations with respect to any use, modification, or sharing of any outputs generated
  by using the Model Materials." Apache-2.0 is *silent* on outputs. Silence is probably
  fine; an express carve-out is better. **This is the clause that makes local distillation
  and synthetic-data generation from Laguna XS 2.1 clean for us.**
- **The Acceptable Use Policy is card language, not licence language.** The S 2.1 card says
  the model "should be used consistently with Poolside's Acceptable Use Policy" `[M]`, but
  the licence text imposes no such condition and contains no field-of-use restriction. Do
  not conflate the two — and do not rely on my reading for anything with money attached.
  `[A]` medium confidence that this distinction survives contact with a lawyer; cheapest
  test is a one-hour review of the two documents side by side before any public release.

### Apache-2.0 / MIT

Apache-2.0's differentiator is the **express patent grant from each contributor** (§3), a
real advantage over MIT's silence for a lab that might publish derivative weights. Its
costs are §4's notice mechanics. MIT is the thinnest of the three: no patent grant, no
retaliation clause. All three are OSI-approved and none restrict field of use.

### The bespoke ones, and the one clause that specifically threatens *this* lab

- **Llama 4 Community License** — separate licence required above 700M MAU at Meta's sole
  discretion, "Built with Llama" attribution, naming requirements on derivatives, AUP
  incorporated `[C]`. Irrelevant to our scale, disqualifying for a clean-provenance story.
- **Kimi Modified MIT** — MIT plus a UI attribution duty above 100M MAU or $20M/month `[C]`.
  Effectively permissive for us.
- **MiniMax Model License** — MIT-shaped, but incorporates a **Prohibited Uses Policy**
  (military use, automated decisions affecting legal rights, etc.) and requires the
  licensee to **indemnify MiniMax** against third-party claims `[C]` (LICENSE-MODEL, HF).
  A use-restricted licence with an indemnity is not OSI-open; treat it as rung 1.
- **NVIDIA Open Model License** — broadly permissive on commercial use, derivatives,
  redistribution, and outputs, but: rights **terminate automatically** if you "bypass,
  disable, reduce the efficacy of, or circumvent any technical limitation, **safety
  guardrail or associated safety guardrail hyperparameter**" `[C]`
  ([NVIDIA Open Model License](https://www.nvidia.com/content/dam/en-zz/Solutions/license-agreements/enterprise-software/nvidia-open-model-license-agreement4-28-2025.pdf)).

That last clause is not abstract for a memory-systems lab. "Alignment Collapse Under KV
Cache Quantization" reports refusal-rate collapse at perplexity deltas small enough that
PPL-only evaluation misses it `[C]` arXiv `2606.09864` (1 Jun 2026), and "The Pitfalls of
KV Cache Compression" finds instruction-dropping and system-prompt leakage under
StreamingLLM/SnapKV/TOVA/H2O/K-Norm `[C]` arXiv `2510.00231` (v2, 2026). An aggressive
Mnemosyne eviction or quantization policy *is* a mechanism that reduces the efficacy of a
safety guardrail, whether or not that was the intent. **Recommendation: run Mnemosyne
policy evaluations against OpenMDW / Apache / MIT models (Laguna XS 2.1, Qwen3.6-35B-A3B,
OLMo 3, gpt-oss-20b) and keep Nemotron out of any published safety-degradation result
until someone competent reads that clause.** `[A]` medium-high confidence the clause
reaches us; cheapest test is a written question to NVIDIA's model-licensing contact.

---

## 4. What rung 2 already bought us — and the three claims it falsified

The strongest empirical argument in this note is local. At rung 2 we get a runnable
`modeling_laguna.py` and a full llama.cpp implementation, and reading them contradicted
the model cards three times.

1. **Router gating.** The S 2.1 card says "token-choice router with **softplus** gating
   over 256 routed experts"; the XS 2.1 card says **sigmoid** gating; the code says
   `routing_scores = torch.sigmoid(router_logits)` at `modeling_laguna.py:183`, with an
   aux-loss-free correction bias at :185 and router-logit softcapping at :181. `[M]`
   CODE_MAP, clones @ `b6d5084fb4a5`. The two official cards disagree with each other.
2. **FP8 KV cache.** The XS 2.1 card advertises "KV cache in FP8." Poolside's own
   llama.cpp branch @ `04b2b72cb540` has **no FP8 anywhere** in `src/`, `common/` or
   `include/`; the quantized-KV mechanism is ordinary block quantization plus a Hadamard
   rotation (`attn_rot_k`, `llama-kv-cache.cpp:319`) enabled by a runtime heuristic and
   disableable by an undocumented env var. `[M]` CODE_MAP. Our own PROVENANCE row for that
   clone repeats the card's FP8 claim and should be read as inherited, not verified.
3. **Uniform head count.** `config.json` advertises `num_attention_heads: 48`;
   `modeling_laguna.py` sources per-layer `num_heads` from
   `config.num_attention_heads_per_layer`. Any KV-cost arithmetic done from the config
   alone is wrong for some layers — `laguna-heads-uniform` is **refuted** in
   `ASSUMPTIONS.md` `[M]` 2026-07-26.

Three for three. The practical rule: **a model card is a marketing artifact with a
technical veneer; the weights-plus-code tier is the first rung where claims become
checkable, and it is a much bigger step up from rung 1 than rung 3 is from rung 2.**

The corollary is what rung 2 *cannot* buy. Laguna's 3:1 global-to-SWA interleave is `[M]`
from `config.layer_types` — but whether 3:1 was *ablated* or *inherited* is unanswerable,
because the ablation was never published. Same for Kimi Linear's 3:1 KDA:MLA
(`2510.26692`) and Qwen3-Next/3.5's roughly 3:1 Gated DeltaNet interleave. The only fully
open hybrid is Olmo Hybrid (`2604.03444`, Apr 2026), and it prescribes no ratio
methodology. **The ratio question is open because of the openness gap, not despite it** —
see `research/memory/hybrid-architectures.md`.

---

## 5. Where the genuinely open questions sit

1. **Ratio and layer-assignment methodology.** Nobody published the sweep. `2507.06457`
   (72 models) finds recall collapses below ~3:1 — a ceiling reading; `2606.15378`
   (13 Jun 2026) argues configurations converge given enough tokens and the
   efficient-attention choice governs *when* long-context ability emerges — an
   emergence-timing reading. Same year, incompatible framings, and the resolution likely
   depends on token budget, which is exactly a small-scale axis.
2. **MLA vs GQA below 1B.** MLA's wins (`2405.04434`) are reported at 100B+ MoE scale. As
   far as the papers track can find, nobody has tested whether the low-rank latent
   survives at 20M–300M. Genuinely available, not a replication.
3. **Attribution in KV compression.** The field reports outcomes, not mechanisms; several
   groups argue most of PyramidKV's gain is SnapKV's observation window rather than the
   per-layer allocation it claims credit for. `2605.08234` argues task accuracy alone
   cannot tell you *why* a selector worked. Top-ranked item in
   `research/memory/open-problems-ranked.md`.
4. **Whether small-proxy methodology transfers.** Poolside's AutoMixer trained "a swarm of
   proxy models" and validated on **a 3B model for 1.5T tokens** `[C]` (technical report
   §3.2.3). That is the only public statement from a frontier lab about proxy scale for
   mixture search — and it is an order of magnitude above our budget. Our
   `ablation-scale-sufficient` assumption is still `untested`.
5. **Corpus provenance.** No frontier open-weight model names its corpus. Laguna discloses
   *proportions* (raw code 30.6%, web 25.2%, synthetic/code-text 25.4%, math 9.0%,
   knowledge 6.6%, instruction-like 1.4%, academic 1.1%, books 0.7%; ~27T unique-token
   pool, >30T tokens trained) `[C]` (technical report Table 4, §3.2) — which is
   unusually generous for rung 3 and still not a dataset. The Common Pile / Comma line
   (`2506.05209`) is the only serious counterexample, and it exists precisely because it
   restricted itself to openly-licensed text.

---

## 6. Contested — presented as contested

- **Does "open weights" earn the word open?** OSI's OSAID 1.0 requires *data information*,
  not data; SFC and others argue that is open-washing. OSI has a 1.1/2.0 process running
  through Q4 2026 to address the data compromise `[C]`. Viseur and Jullien's five-cluster
  taxonomy — open washing / easy access / open weight / open science / open source — is
  the cleanest academic framing of the split `[C]`. The European Open Source AI Index
  scores 150+ systems on 14 openness parameters and explicitly hunts openwashing `[C]`
  ([osai-index.eu](https://osai-index.eu/), updated 3 Jul 2026). No consensus.
- **Does openness help or hurt safety?** `2604.17413` (Apr 2026) argues restricting access
  may undermine the safety it seeks to protect; `2606.19890` (Jun 2026) reviewed 37
  open-weight model families released 2025–Apr 2026 and found **one** met a comprehensive
  proportional-evaluation standard, most met none; the International AI Safety Report 2026
  (`2602.21012`) sits between. Live dispute, and a policy fight: a 24 Jul 2026 industry
  letter from ~25 organisations including NVIDIA, Microsoft, Meta, IBM, Hugging Face and
  Mistral argues US leadership depends on an open ecosystem `[C]`
  ([Open Weights and American AI Leadership](https://images.nvidia.com/pdf/Open-Weights-and-American-AI-Leadership.pdf)).
- **Is the gap closing?** Epoch says 3 months (Oct 2025) → 4 months (May 2026), 6 months
  under a stricter comparison rule, with both known biases pointing toward
  under-statement `[C]`. Vendor and press narratives say the opposite. Believe the
  index over the narrative, and note the index is one lab's composite.
- **Supply risk on the open frontier.** Four of the five leading open-weight families are
  Chinese. MOFCOM is reported (FT and Reuters, 21 Jul 2026) to be consulting Alibaba,
  ByteDance and Zhipu on export controls covering model weights and training-data
  transfer, with a tiered regime floated up to a possible ban on public release of the
  most capable models `[C]`. **Nothing is decided.** But `[A]` medium confidence that this
  is a real risk to the reference set within 12 months; the cheapest mitigation is
  free — mirror the configs and cards we depend on now, which `fetch_reference.sh`
  already does, and keep PROVENANCE revisions pinned.

---

## 7. Hooks into the memory track

- **`memory-taxonomy.md` / reconstructibility.** Openness tier and reconstructibility are
  the same kind of question asked at different timescales. The KV cache is reconstructible
  by recompute in milliseconds; the weights are "reconstructible" only by re-running a
  $10M training job you cannot reproduce bit-exactly. Rung-4 releases are the only ones
  where the outer reconstruction is even attemptable.
- **`kv-serving-hierarchy.md` / CODE_MAP.** Everything we know about real KV management —
  vLLM's block table, SGLang's radix tree, Mooncake's lease-based tiering, FlashInfer's
  CSR page table — comes from **Apache-2.0 / MIT serving code**, not from model releases.
  The serving layer is genuinely rung 4 and the model layer is genuinely rung 2. Mnemosyne's
  separability requirement is well served by that: the interface it must satisfy is
  documented in open code even though the models it will manage are not.
- **`hybrid-architectures.md`.** Laguna's 3:1 is `[M]` only because rung 2 included the
  modeling file and a llama.cpp port; and it is *unvalidated* only because rung 4 was
  withheld. That sentence is the whole argument of this note in miniature.
- **`kv-compression-and-eviction.md` + licensing.** The NVIDIA guardrail clause plus
  `2606.09864` means our eviction/quantization work has a licence dependency, not just a
  citation dependency. Choose target models accordingly.
- **Hardware ceiling.** Our fast tier is **≥62 GiB at ~200 GB/s** `[M]`
  (`notebook/uma-carveout-controls-fast-tier.md`, 2026-07-26, single run per arm), with
  single tensors **≥32 GiB refuted** as safe `[M]`. That decides which open models are
  runnable here at all: Laguna XS 2.1 (33B-A3B) in FP8/INT4 fits comfortably; Laguna S 2.1
  BF16 needs ~236 GB by its own card `[M]` and does not. Rung-4 models at 3B–7B (Instella,
  OLMo 3-7B, Comma, Apertus-8B) all fit, which is convenient given they are also the only
  ones we can run counterfactuals against.
- **Housekeeping.** The Laguna M.1/XS.2 technical report (poolside.ai PDF, 25 May 2026) is
  now a load-bearing source and is **not** in `PROVENANCE.md`. Add a row.

---

## Open questions

Testable at 20M–300M params on one GPU, ≥62 GiB fast tier, no collectives.

1. **Was 3:1 ablated or inherited?** Take Laguna's `layer_types` list verbatim (`[M]`,
   pinned) and sweep GSSS / all-global / all-sliding / 1:1 at matched params and tokens on
   an openly-licensed corpus (Common Pile or FineWeb-Edu), scoring MQAR plus a RULER-style
   synthetic battery rather than perplexity. Directly tests the `2507.06457` "ceiling"
   reading against the `2606.15378` "emergence-timing" reading, which differ most at small
   token budgets. Riskiest part: 20M–300M may be below the scale where retrieval heads form
   at all — which is itself a publishable negative.
2. **Does proxy-scale ranking transfer downward?** Rank two data mixtures with a
   150M/3B-token proxy and again with a 300M/5B-token proxy; measure rank agreement. This
   is the cheapest available attack on `ablation-scale-sufficient` (currently `untested`),
   and it is the one methodological claim a frontier lab has actually published a number
   for (3B / 1.5T).
3. **What is our reproducibility floor?** Two fixed-seed runs of the same tiny recipe on
   gfx1151; report loss-curve divergence. Below that floor, any "we reproduced X" claim
   from this machine is noise. Feeds the Hardware Validation Gate's determinism item,
   which has not run — and `bf16-numerics-unproven` is still `untested`, so no result from
   this machine counts as evidence yet.
4. **Does the openness tier change what an eviction policy looks like?** Run the same
   Mnemosyne policy against a rung-2 model (Laguna XS 2.1) and a rung-4 model (OLMo 3-7B),
   and check whether access to intermediate checkpoints changes the *attribution* story —
   i.e. can you distinguish "the policy helped" from "the policy exploited a training
   artifact" only when you can see training?
5. **Is licence-clean synthetic data actually clean?** OpenMDW-1.1's output clause plus a
   locally-run Laguna XS 2.1 gives an unencumbered generation path on paper. Generate a
   small set, document the chain (licence text @ pinned revision → output clause → no AUP
   in the licence body), and get one legal read before anything is published. Cheap, and it
   unblocks a whole class of experiments if it holds.

**Riskiest assumption in this note:** that reading licence text carefully is a substitute
for legal advice. It is not. Everything in §3 is a careful reading of primary documents by
someone who is not a lawyer, and the OpenMDW-1.1 output clause and the NVIDIA guardrail
clause are the two places where being wrong would be expensive.

---

## Sources

**Primary artifacts held locally** (pinned revisions in `research/reference/PROVENANCE.md`,
fetched 2026-07-26):
`models/laguna-s/LICENSE.md`, `README.md`, `config.json`, `modeling_laguna.py` @ `b0a9fd7c850e`;
`models/laguna-xs/README.md`, `config.json` @ `205dc65dd4bd`;
`models/gpt-oss-20b/LICENSE` (Apache-2.0), `USAGE_POLICY`;
`models/qwen3-next/LICENSE` (Apache-2.0);
`models/nemotron-nano/README.md` (nvidia-open-model-license);
`models/kimi-linear-model/README.md` (MIT);
`architecture/transformers` @ `b6d5084fb4a5`; `architecture/llama-cpp-laguna` @ `04b2b72cb540`;
`research/reference/CODE_MAP.md`; `ASSUMPTIONS.md`; `notebook/uma-carveout-controls-fast-tier.md`.

**arXiv** (every id below resolved against the live arXiv API on 2026-07-26):
`2403.13784` Model Openness Framework ·
`2405.15802` Columbia Convening on Openness in AI ·
`2502.18505` Transparency and Accessibility of SoTA LLMs ·
`2506.05209` The Common Pile v0.1 ·
`2506.01732` Common Corpus ·
`2508.10925` gpt-oss-120b & gpt-oss-20b Model Card ·
`2509.14233` Apertus ·
`2511.10628` Instella ·
`2512.13961` Olmo 3 ·
`2312.06550` LLM360 · `2501.07124` LLM360 K2 ·
`2602.21012` International AI Safety Report 2026 ·
`2604.17413` The Open-Weight Paradox ·
`2606.19890` Open Weight AI Models Require Proportional Evaluation Approaches ·
`2606.26099` Benchmarking Open-Weight Foundation Models for Global AI Technical Governance ·
`2606.19348` DeepSeek-V4 · `2512.02556` DeepSeek-V3.2 · `2405.04434` DeepSeek-V2 ·
`2510.26692` Kimi Linear · `2605.26494` MiniMax-M2 Series ·
`2504.03624` Nemotron-H · `2604.03444` Olmo Hybrid ·
`2507.06457` A Systematic Analysis of Hybrid Linear Attention ·
`2606.15378` Rethinking the Role of Efficient Attention in Hybrid Architectures ·
`2606.09864` Alignment Collapse Under KV Cache Quantization ·
`2510.00231` The Pitfalls of KV Cache Compression ·
`2605.08234` When Does Value-Aware KV Eviction Help? ·
`2607.02574` From Tensor Buffer to Distributed Memory Hierarchy.

**Licences and licence bodies:**
[OpenMDW](https://openmdw.ai/) and [OpenMDW/OpenMDW on GitHub](https://github.com/OpenMDW/openmdw) ·
[Linux Foundation, OpenMDW-1.1 release + NVIDIA adoption, 28 May 2026](https://www.linuxfoundation.org/press/linux-foundation-releases-openmdw-1.1-nvidia-adopts-openmdw-for-cosmos-isaac-gr00t-ising-and-nemotron-ai-model-families) ·
[NVIDIA Open Model License Agreement (28 Apr 2025 PDF)](https://www.nvidia.com/content/dam/en-zz/Solutions/license-agreements/enterprise-software/nvidia-open-model-license-agreement4-28-2025.pdf) ·
[MiniMax-M2.5 LICENSE-MODEL](https://huggingface.co/MiniMaxAI/MiniMax-M2.5/blob/main/LICENSE-MODEL).

**Vendor and lab primary sources:**
[Laguna M.1/XS.2 Technical Report, 25 May 2026 (PDF)](https://poolside.ai/assets/laguna/laguna-m1-xs2-technical-report.pdf) ·
[Introducing Laguna S 2.1](https://poolside.ai/blog/introducing-laguna-s-2-1) ·
[Ai2, Olmo 3, 20 Nov 2025](https://allenai.org/blog/olmo3) ·
[NVIDIA, Inside Nemotron 3](https://developer.nvidia.com/blog/inside-nvidia-nemotron-3-techniques-tools-and-data-that-make-it-efficient-and-accurate/) ·
[Google, Gemma 4, 2 Apr 2026](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) ·
[Hugging Face, SmolLM3](https://huggingface.co/blog/smollm3) and
[The Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) ·
[AMD Instella (GitHub)](https://github.com/AMD-AGI/Instella) ·
[EleutherAI, The Common Pile v0.1](https://blog.eleuther.ai/common-pile/) ·
[Open Weights and American AI Leadership, 24 Jul 2026 (PDF)](https://images.nvidia.com/pdf/Open-Weights-and-American-AI-Leadership.pdf).

**Measurement, indices and policy:**
[Epoch AI, open vs closed ECI gap, 29 May 2026](https://epoch.ai/data-insights/open-closed-eci-gap) ·
[European Open Source AI Index](https://osai-index.eu/) (14 parameters, database updated 3 Jul 2026) ·
[Model Openness Tool](https://isitopen.ai/) ·
[EC, Guidelines for providers of general-purpose AI models](https://digital-strategy.ec.europa.eu/en/policies/guidelines-gpai-providers) ·
[EU AI Act GPAI guidelines overview](https://artificialintelligenceact.eu/gpai-guidelines-overview/).

**Dated press used as leads, not evidence** (all read 2026-07-26):
[VentureBeat on Laguna S 2.1, 21 Jul 2026](https://venturebeat.com/infrastructure/poolside-drops-laguna-s-2-1-an-open-weight-coding-model-that-beats-rivals-10x-its-size) ·
[Simon Willison on GLM-5.2, 17 Jun 2026](https://simonwillison.net/2026/jun/17/glm-52/) ·
[Simon Willison on Inkling, 16 Jul 2026](https://simonwillison.net/2026/Jul/16/inkling/) ·
[DeepSeek V4 preview release notes, 24 Apr 2026](https://api-docs.deepseek.com/news/news260424/) ·
[Reuters/FT-derived coverage of MOFCOM AI export-control consultation, 21–22 Jul 2026](https://www.explainx.ai/blog/china-overseas-ai-model-restrictions-reuters-july-2026) ·
[LWN on the OSI Open Source AI Definition](https://lwn.net/Articles/995159/).
