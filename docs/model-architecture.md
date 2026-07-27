# Model architecture and system layouts

Every diagram in this document is **generated from the artifacts, not drawn from
memory** — each element traces to a config key, a `file:line` in `research/reference/`,
or a row in `ASSUMPTIONS.md`. Across the eleven diagrams that is **336 grounding
references**.

Sources live in `docs/diagrams/*.mmd` as Mermaid, not as binary drawings, so they diff,
review in a pull request, and render on GitHub. `scripts/validate_diagrams.py` renders
every one and **fails the build on any that does not** — a broken diagram does not look
broken in source, it looks like a diagram, and only fails as a grey error box in front of
a reader. It found three such blocks already committed in `v0.7.0`.

```
python scripts/validate_diagrams.py                      # render-check every diagram
python scripts/validate_diagrams.py --svg docs/diagrams/rendered   # export SVG
```

Use `--svg` if you want files to open in Visio, Lucidchart or Figma. The Mermaid text
stays the source of truth; the SVG is a build artifact.

**A standing caveat.** Several diagrams carry measured numbers from `ASSUMPTIONS.md`.
Those are single-run figures on an instrument whose numerics are unproven — the Hardware
Validation Gate has not run. Read them as sizing, not as results.

---


# Model architecture

## The reference stack: 48 layers, two types

```mermaid
flowchart TB
  IN["input_ids"]
  EMB["embed_tokens<br/>vocab 100352 to d_model 3072"]
  NORM["final RMSNorm<br/>rms_norm_eps 1e-6"]
  HEAD["lm_head to 100352 logits<br/>tie_word_embeddings false"]

  IN --> EMB

  subgraph UNIT1["Unit 1 of 12 -- layers 0 to 3 -- the only unit that is not identical"]
    direction TB
    L0["layer 0 -- GLOBAL attention<br/>48 query heads / 8 KV heads -- GQA group 6<br/>DENSE MLP: the one layer with no router"]
    L123["layers 1, 2, 3 -- SLIDING attention<br/>72 query heads / 8 KV heads -- GQA group 9<br/>sliding_window 512 -- MoE"]
    L0 --> L123
  end

  subgraph UNITN["Units 2 to 12 -- layers 4 to 47 -- this same shape, 11 more times"]
    direction TB
    G["G -- layers 4, 8, 12 ... 44 -- GLOBAL<br/>48 query heads / 8 KV heads -- GQA group 6<br/>attends every position, up to 1048576<br/>YaRN RoPE, theta 500000, on 64 of 128 head dims"]
    S["S S S -- the three layers after each G -- SLIDING<br/>72 query heads / 8 KV heads -- GQA group 9<br/>sees only the most recent 512 tokens<br/>plain RoPE, theta 10000, on all 128 head dims"]
    G --> S
  end

  EMB --> L0
  L123 -->|"then units 2 to 12"| G
  S -->|"out of layer 47"| NORM
  NORM --> HEAD

  subgraph MLPSLOT["Reference panel -- what fills the MLP slot"]
    direction TB
    DENSE["layer 0 only -- LagunaMLP, intermediate_size 12288<br/>selected because mlp_only_layers names layer 0"]
    MOE["layers 1 to 47 -- LagunaSparseMoeBlock<br/>256 experts, top 10 chosen by sigmoid score, moe_intermediate_size 1024<br/>plus an always-on shared expert of width 1024"]
  end

  subgraph KVCOST["Reference panel -- what the interleave buys, bf16 KV cache"]
    direction TB
    KVPER["Per layer, per token: 2 x 8 KV heads x 128 head_dim x 2 bytes = 4.0 KiB<br/>identical on every layer -- query-head count never enters this product"]
    KVFLAT["If all 48 layers were global: 192.0 KiB per token = 24.0 GiB at 128k context"]
    KVREAL["As actually built: 12 global layers grow with context = 6.0 GiB at 128k<br/>36 sliding layers are pinned at 512 tokens = 2.0 MiB each, 72 MiB total<br/>about 6.1 GiB -- a 4x cut. Arithmetic from config, not a measurement."]
    KVFIT["Budget against the measured fast tier: at least 62 GiB at about 200 GB/s"]
    KVPER --> KVFLAT --> KVREAL --> KVFIT
  end
```

Laguna-S 2.1 is 48 decoder layers, but it is only two layer types plus one exception, so the diagram draws the repeating unit instead of 48 boxes. `layer_types` in the shipped config is a strict GSSS cycle — one global layer then three sliding — repeated twelve times, giving 12 full and 36 sliding. Layer 0 is the exception on a second axis: it is the sole entry in `mlp_only_layers`, so the one layer that attends globally is also the one with a dense MLP and no router. The remaining 47 all carry the 256-expert, top-10 MoE block.

The non-obvious thing is which number varies. Query heads change with layer type — 48 on global, 72 on sliding — but `num_key_value_heads` is a scalar 8 with no per-layer override, so KV cache cost is *identical* on every layer at 4.0 KiB per token in bf16. What actually varies is the GQA group size, 6 on global and 9 on sliding: a compute and arithmetic-intensity difference, not a capacity one. Cache savings come entirely from the 512-token window, and they are large: roughly 6.1 GiB at 128k context against 24.0 GiB if every layer were global.

That looks exactly like a two-tier storage hierarchy, and the sizing math is capacity planning. Three places the analogy breaks. There is no promotion, demotion, or miss path — a layer is bound to its tier by index at construction. Discarding out-of-window tokens is not a gamble with a hit-rate cost; the mask makes them unreadable, so it is lossless by construction. And the tiers are not numerically interchangeable: global layers use YaRN RoPE at theta 500000 over half the head dims, sliding layers plain RoPE at theta 10000 over all of them. You cannot widen the windows to test long context.

**What this diagram omits.** Layer internals are omitted entirely: QK-norm before RoPE, the per-head softplus output gate (config gating: per-head), the two RMSNorms and the two residual adds, router-logit softcapping, the aux-loss-free e_score_correction_bias, and moe_routed_scaling_factor 2.5. Those belong in a separate layer-internals diagram rather than crammed into this one. The ~6.1 GiB figure (12 global layers growing to 6.0 GiB at 128k, plus 36 windowed layers at 2.0 MiB each) is arithmetic over measured config values, not a measured allocation. The label says so. Only the 24.0 GiB all-global figure (ASSUMPTIONS row kv-per-token-laguna) and the >=62 GiB fast tier (row gpu-fast-tier-size) carry register entries. The windowed-residency claim assumes the serving stack actually caps the sliding layers' cache at 512 tokens. llama.cpp demonstrably does (CODE_MAP.md:61, two separate llama_kv_cache objects). Whether HF DynamicCache does was not checked, and the diagram does not distinguish the two. Parameter counts are not shown. 118B total / 8.5B active comes from the model card, not from anything derived here, so it was left out rather than restated. Masks and position embeddings are built once per layer *type* per forward and shared across all layers of that type (modeling_laguna.py:565-582). The diagram shows the type split but not that sharing. The diagram asserts KV cost is uniform across layers. This directly contradicts CODE_MAP.md:54 ("any KV-cost calculation done from the config alone is wrong for some layers"). The diagram follows the source and ASSUMPTIONS.md row laguna-heads-uniform, which explicitly corrects that claim. CODE_MAP is generated by scripts/generate_code_map.py and was not edited here; flagged back to its owner. The GSSS pattern is drawn as if it were structure. It is data: one list lookup at construction (modeling_laguna.py:365). That is stated in the prose but a diagram inherently makes it look architectural. This is the shipped Laguna-S 2.1 config at revision b0a9fd7c850e. CODE_MAP.md:73 records 48-vs-64 query heads for XS.2, so head counts are variant-specific and this diagram does not generalise across the family.

*Source: [`docs/diagrams/laguna-decoder-stack.mmd`](diagrams/laguna-decoder-stack.mmd) — 33 grounding references.*

---

## Inside one block

```mermaid
flowchart TD
    IN["hidden_states in : [B, T, 3072]"]

    subgraph attnpath["Attention sublayer : LagunaDecoderLayer.forward, lines 450-462"]
        LN1["input_layernorm : RMSNorm 3072, eps 1e-6"]
        QPROJ["q_proj, no bias : 3072 -> n_heads x 128"]
        KVPROJ["k_proj / v_proj, no bias : 3072 -> 8 KV heads x 128, uniform across all layers"]
        QKNORM["q_norm / k_norm : RMSNorm over head_dim 128, applied before RoPE"]
        ROPE["apply_rotary_pos_emb : cos/sin built once per layer TYPE, not per layer"]
        CACHE[("past_key_values.update : 192.0 KiB per token all-layers = 24.0 GiB at 128k ctx; windowed layers hold 512 tokens, not ctx")]
        SDPA["attention_interface : GQA, scale = 128^-0.5"]
        GPROJ["g_proj -> softplus : one scalar per head"]
        GMUL["per-head output gate : scales all 128 dims of each head"]
        OPROJ["o_proj : n_heads x 128 -> 3072"]
    end

    RES1(["residual add"])

    subgraph ffnpath["FFN sublayer : chosen by config.mlp_layer_types, line 433"]
        LN2["post_attention_layernorm : RMSNorm 3072"]
        DENSE["LagunaMLP dense : SwiGLU, intermediate 12288"]
        ROUTER["LagunaTopKRouter : sigmoid of W.h, independent scores, no softmax budget"]
        SELECT["+ e_score_correction_bias, then top-10 of 256 : bias shifts SELECTION only, aux loss coef 0.0"]
        EXPERTS["10 routed experts : SwiGLU width 1024, weighted by normalised sigmoid scores"]
        SHARED["shared expert : SwiGLU width 1024, runs for every token"]
        SCALE["x 2.5 : moe_routed_scaling_factor"]
        SUMN["routed + shared"]
    end

    RES2(["residual add"])
    OUT["hidden_states out : [B, T, 3072]"]

    subgraph layertype["Chosen by a lookup in config.layer_types, line 365 : the ONLY per-layer difference"]
        FULL["full_attention, 12 of 48 layers : 48 q heads / 8 kv, GQA group 6; YaRN theta 500000 over 64 of 128 dims"]
        SWA["sliding_attention, 36 of 48 layers : 72 q heads / 8 kv, GQA group 9; plain RoPE theta 10000 over all 128 dims"]
    end

    IN --> LN1
    IN -->|"residual, carried untouched"| RES1
    LN1 --> QPROJ
    LN1 --> KVPROJ
    LN1 -->|"gate reads the LAYER INPUT, not the attention output"| GPROJ
    QPROJ -->|"Q"| QKNORM
    KVPROJ -->|"K only"| QKNORM
    KVPROJ -->|"V: no norm, no RoPE"| CACHE
    QKNORM --> ROPE
    ROPE -->|"K"| CACHE
    ROPE -->|"Q"| SDPA
    CACHE -->|"K, V for every readable position"| SDPA
    SDPA --> GMUL
    GPROJ --> GMUL
    GMUL --> OPROJ
    OPROJ --> RES1

    RES1 --> LN2
    RES1 -->|"residual"| RES2
    LN2 -->|"dense: layer 0 only"| DENSE
    LN2 -->|"sparse: 47 of 48 layers"| ROUTER
    LN2 -->|"sparse layers only, unrouted"| SHARED
    ROUTER --> SELECT
    SELECT --> EXPERTS
    EXPERTS --> SCALE
    SCALE --> SUMN
    SHARED --> SUMN
    DENSE --> RES2
    SUMN --> RES2
    RES2 --> OUT

    FULL -.->|"sliding_window = None; full causal mask"| SDPA
    SWA -.->|"sliding_window = 512; windowed causal mask"| SDPA
    FULL -.->|"rope_parameters.full_attention"| ROPE
    SWA -.->|"rope_parameters.sliding_attention"| ROPE
```

What you are looking at is one of the 48 layers, drawn end to end. Two sublayers, each pre-normed, each ending in an addition back onto the residual stream — the only path that runs the full depth of the model. Everything except the two boxes at the bottom right is identical in every layer; those two are selected by an index lookup into a config list at construction time, not by any runtime condition.

The non-obvious element is the per-head gate. `g_proj` reads the *normed layer input*, in parallel with the QKV projections — not the attention output — so the scalar that scales each head's 128 output dimensions is computed without ever seeing what attention retrieved. It is a learned volume knob per head, applied before `o_proj` mixes the heads together, which lets a head be muted for a given token before its contribution reaches the residual stream. This is not part of the standard decoder recipe, and the upstream code carries no comment explaining it.

Where the systems analogy breaks: the KV cylinder looks like a write-back cache and is not one. There is no miss path, no backing store and no promotion, and on a sliding layer out-of-window keys are architecturally unreadable, so discarding them is lossless rather than a gamble. Nor are the two layer types the same block with a different mask — they have different query-head counts (48 vs 72), different GQA group sizes (6 vs 9) and different positional encodings (YaRN at θ=500000 over half the head dimensions, versus plain RoPE at θ=10000 over all of them). Widening the sliding window to test long context is therefore not a configuration change: those layers were never trained with positional encoding that reaches past 512 tokens.

**What this diagram omits.** Scope is one layer. The embedding table, the 48-layer stack, the final norm and lm_head are out of frame, as is the 118B-total / 8.5B-active split — none of that is visible in a single block. The routed-expert box hides a Python for-loop over hit experts (modeling_laguna.py:219-229) that gathers tokens per expert and index_add_s results back. That is a reference implementation, not the dispatch a real deployment uses; drawing it would teach the wrong cost model. GAP FLAGGED, NOT PAPERED OVER: router-logit softcapping exists in code (modeling_laguna.py:180-181) and CODE_MAP.md:52 describes it as an active guard against expert collapse, but the shipped config sets moe_router_logit_softcapping to 0.0, so the branch never executes. Omitted from the diagram deliberately. CODE_MAP is documentation (class 3) and its Laguna section should say the guard is present but disabled at the shipped config. GAP FLAGGED: the '24.0 GiB at 128k' figure is all-layers-hold-full-context arithmetic from ASSUMPTIONS row kv-per-token-laguna. True residency is far lower because 36 of 48 layers cap at 512 tokens, and no register row currently carries that residency number. The label says 'windowed layers hold 512 tokens, not ctx' rather than inventing one. Someone should compute it and add a row. Attention is a single box. No eager/sdpa/flash distinction, no mask construction, no softmax internals, and no attn_weights return path. On this hardware that abstraction hides a real hazard: ASSUMPTIONS row sdpa-is-memory-efficient records that the default path retains the B*nh*T*T score matrix (147.2 bytes/T-squared) unless TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 is set. GQA fan-out is stated in labels (8 KV heads serving 48 or 72 query heads) rather than drawn. Drawing the fan-out would have cost roughly ten nodes for one fact. Dropout, gradient checkpointing, and the fp32 upcast inside RMSNorm and inside the router are not shown as separate stages. This is the shipped Laguna S 2.1 config read at revision b0a9fd7c850e. It is the reference architecture, not Proteus's config surface, and should not be read as a target design.

*Source: [`docs/diagrams/laguna-decoder-block.mmd`](diagrams/laguna-decoder-block.mmd) — 44 grounding references.*

---

## The MoE path: sigmoid routing and a bias nobody advertises

```mermaid
flowchart TD
    H["hidden_states, flattened to B*T by 3072<br/>hidden_size = 3072"]

    subgraph ROUTER["LagunaTopKRouter - modeling_laguna.py lines 163-191"]
        LOGITS["router linear, cast to fp32<br/>weight 256 by 3072, line 178"]
        CAP["tanh softcap, line 181<br/>IMPLEMENTED BUT SHIPPED OFF<br/>moe_router_logit_softcapping = 0.0"]
        SIG["sigmoid, line 183<br/>routing_scores: 256 independent scores<br/>NOT softmax - no shared probability budget"]
        SEL["add e_score_correction_bias, line 185<br/>256 values, requires_grad False<br/>aux-loss-free: router_aux_loss_coef = 0.0"]
        TOPK["topk k = 10 of 256, line 186<br/>returns indices only, scores discarded"]
        GATHER["gather routing_scores at selected_experts<br/>line 187 - reads the PRE-BIAS values"]
        NORM["divide by their sum, line 188<br/>the 10 routing_weights sum to 1"]
    end

    subgraph EXPERTS["LagunaExperts - one repeating unit; 10 of 256 fire per token"]
        EX["expert e: gate_up_proj 2048 by 3072, chunk into gate and up<br/>SwiGLU, act_fn of gate times up, lines 225-226"]
        DOWN["down_proj back to 3072, line 227<br/>moe_intermediate_size = 1024"]
        WEIGHT["scale by this token-slot routing_weight, line 228"]
        ACC["index_add into final_hidden_states, line 229"]
    end

    SHARED["shared expert: LagunaMLP, line 245<br/>shared_expert_intermediate_size = 1024<br/>ALWAYS ON - no router, no gate, every token"]

    SCALE["times moe_routed_scaling_factor = 2.5, line 250"]
    ADD["add shared_output, line 251"]
    OUT["block output - 47 of 48 layers take this path<br/>layer 0 is a plain dense LagunaMLP, line 433"]

    H --> LOGITS
    H --> SHARED
    H --> EX
    LOGITS -.->|"bypassed while softcapping is 0.0"| CAP
    CAP -.-> SIG
    LOGITS --> SIG
    SIG --> SEL
    SEL --> TOPK
    TOPK -->|"which experts"| GATHER
    SIG -->|"the values - bias steers selection, never the weights"| GATHER
    GATHER --> NORM
    TOPK -->|"expert ids"| EX
    NORM -->|"combine weights"| WEIGHT
    EX --> DOWN
    DOWN --> WEIGHT
    WEIGHT --> ACC
    ACC --> SCALE
    SCALE --> ADD
    SHARED --> ADD
    ADD --> OUT
```

The diagram traces one token's hidden state through a single sparse Laguna MoE layer — 47 of the 48 layers, since `mlp_layer_types[0]` is `dense`. Two paths leave the same input: a shared expert that runs unconditionally on every token, and a router that selects 10 of 256.

Follow the two edges converging on `gather`. `topk` supplies the indices, but the values come from the sigmoid output *before* `e_score_correction_bias` was added (`modeling_laguna.py:185-187`). That is the entire aux-loss-free load-balancing trick: the bias shifts *which* experts win and never touches *how much* their output counts. `router_aux_loss_coef` is `0.0` in the shipped config, so this bias is the only balancing mechanism in the model — and it carries `requires_grad=False`, meaning it is not learned by backprop but written out-of-band by the trainer and frozen at inference.

The softcap on line 181 is real code behind `if self.router_logit_softcapping > 0.0`. The shipped value is `0.0`, so it never executes. Worth knowing it exists before concluding Laguna has no guard against router runaway. Sigmoid rather than softmax is the other structural choice: experts are scored independently instead of competing for a fixed probability budget, and the normalise on line 188 restores sum-to-one only across the 10 survivors.

Where the load-balancer analogy breaks: this balancer has no queue depth, no health check, and no overflow. Every selected expert runs — there is no capacity factor and no dropped token in this implementation. And imbalance is not a latency problem you can drain. During training an under-used expert receives fewer gradients and permanently learns less, so imbalance is self-reinforcing rather than transient.

**What this diagram omits.** Depicts one sparse MoE layer only. The 48-layer stack, attention, RMSNorm and residuals are out of scope; the diagram's only concession to the stack is the OUT node noting 47 of 48 layers take this path. LagunaExperts is decorated @use_experts_implementation (modeling_laguna.py:194). The per-expert loop drawn as the repeating unit is the readable reference implementation, not necessarily the kernel that runs. Shapes are per-token after the flatten at modeling_laguna.py:244. The batch/sequence collapse and the reshape back at :253 are not drawn. config.json keys norm_topk_prob, decoder_sparse_step and mlp_only_layers are present in the artifact but inert - dropped at modular_laguna.py:125-128. The diagram silently uses the live key (mlp_layer_types) and does not show the dead ones. No measured numbers: nothing in ASSUMPTIONS.md measures MoE routing on gfx1151. Every figure in the diagram is a config-file or source read, not an [M]. Inference path only. The rule that updates e_score_correction_bias during training is not in this file, so the mechanism that actually balances load is named but not drawn. moe_apply_router_weight_on_input is false in the shipped config and raises NotImplementedError if set (configuration_laguna.py:154-156); the alternative weighting path is not drawn. The 2.5 routed-scaling factor is drawn where the code applies it - after combine, before the shared-expert add - but the diagram does not explain why routed and shared outputs are on different scales.

*Source: [`docs/diagrams/laguna-moe-routing.mmd`](diagrams/laguna-moe-routing.mmd) — 39 grounding references.*

---

## MHA, GQA, MQA, MLA: one factor, four values

```mermaid
flowchart TB
    FORMULA["KV bytes per token = 2 x L x H_kv x d x b<br/>Laguna S 2.1 constants: L = 48, d = 128, b = 2 bytes for bf16<br/>n_q, hidden_size, parameter count, expert count and vocab do not appear.<br/>A 118B-A8.5B MoE and a 300M dense model with the same four numbers<br/>cost exactly the same per token."]

    subgraph VAR_MHA["MHA: H_kv = n_q = 48, G = 1"]
        QMHA["48 query heads, q_0 to q_47"]
        KMHA["48 KV head pairs<br/>K_i and V_i, 128 wide each"]
        BMHA["2 x 48 x 48 x 128 x 2 B<br/>= 1152 KiB per token<br/>baseline, 1.00x"]
        QMHA -->|"one private KV head each"| KMHA --> BMHA
    end

    subgraph VAR_GQA["GQA-8: H_kv = 8, G = 6. What Laguna ships."]
        QGQA["48 query heads, partitioned into 8 groups of 6"]
        KGQA["8 KV head pairs<br/>group g serves q_6g through q_6g+5<br/>num_key_value_groups = n_q // H_kv"]
        BGQA["2 x 48 x 8 x 128 x 2 B<br/>= 192 KiB per token<br/>6x smaller than MHA"]
        QGQA -->|"6 query heads share 1 KV head"| KGQA --> BGQA
    end

    subgraph VAR_MQA["MQA: H_kv = 1, G = 48"]
        QMQA["48 query heads, all in one group"]
        KMQA["1 KV head pair<br/>shared by every query head"]
        BMQA["2 x 48 x 1 x 128 x 2 B<br/>= 24 KiB per token<br/>48x smaller than MHA"]
        QMQA -->|"48 query heads share 1 KV head"| KMQA --> BMQA
    end

    subgraph VAR_MLA["MLA: the 2 x H_kv x d term is replaced, not resized"]
        QMLA["every query head reads the same entry<br/>per-head K and V are reconstructed by kv_b_proj"]
        KMLA["1 latent per token, shared by all heads<br/>d_c = 512 plus d_r = 64 decoupled rope key = 576 wide<br/>Kimi Linear ranks, dropped into Laguna's 48 layers"]
        BMLA["576 x 48 x 2 B<br/>= 54 KiB per token<br/>21x smaller than MHA, 3.6x smaller than GQA-8"]
        QMLA -->|"no per-head KV at all"| KMLA --> BMLA
    end

    FORMULA --> QMHA
    FORMULA --> QGQA
    FORMULA --> QMQA
    FORMULA -.->|"formula no longer applies"| QMLA

    UNIFORM["Shape note: all four priced at n_q = 48 on every layer, so the columns are comparable.<br/>Laguna actually runs 72 query heads on its 36 sliding layers, so a true MHA Laguna<br/>is 1584 KiB per token, 8.25x the shipped GQA-8.<br/>Query heads set G, hence decode speed. They never set KV bytes."]

    WINDOW["The window edits a different factor: how many tokens are retained,<br/>not how wide each token's entry is. Laguna keeps 12 full layers plus<br/>36 sliding layers at w = 512, so 128k context costs 6.07 GiB, not 24.0 GiB.<br/>Both axes multiply cleanly; they answer different questions."]

    GOTCHA["MEASURED GOTCHA: the HF reference caches the EXPANDED form.<br/>modeling_kimi.py:401 runs kv_b_proj BEFORE modeling_kimi.py:413 calls past_key_values.update,<br/>so the cache holds 32 x 128 nope + 32 x 64 rope for K plus 32 x 128 for V<br/>= 10240 elements = 20 KiB per layer per token.<br/>17.8x larger than the latent and 5x larger than GQA-8's 4 KiB per layer.<br/>The advertised win lives in the serving engine's absorb trick, not in the model class."]

    BMHA --> UNIFORM
    BGQA --> WINDOW
    BMLA --> GOTCHA

    classDef shipped fill:#e8f4ea,stroke:#2f7d3f,stroke-width:2px,color:#14301c
    classDef warn fill:#fdeceb,stroke:#b3352b,stroke-width:2px,color:#3d0f0b
    classDef note fill:#f4f4f4,stroke:#8a8a8a,color:#222222
    class BGQA shipped
    class GOTCHA warn
    class UNIFORM,WINDOW note
```

Four columns, one product. The formula at the top is the entire KV cost model: `2` (one key, one value) x `L` layers x `H_kv` key/value heads x `d` head width x `b` bytes per element. MHA, GQA and MQA are three values of a single integer in it — `H_kv` — and MLA is the one variant that replaces the term rather than resizing it, caching a single 576-wide latent per token instead of per-head keys and values. Every column is priced at Laguna S 2.1's real constants (`L=48`, `d=128`, bf16) with query heads held at 48 so the four are comparable, which is why the shipped GQA-8 column reads exactly 192 KiB/token.

The non-obvious thing is what the product does not contain: query heads, hidden size, parameter count, expert count, vocabulary. Laguna's 118B-A8.5B MoE and a 300M dense model with the same four numbers have identical KV cost per token — so the *small* model enters the KV-dominated regime at shorter contexts, not longer. The same absence is why MQA prices below MLA here: MLA's advertised 93.3% reduction is measured against an MHA baseline, and against GQA-8 at this shape it is 3.6x.

The systems analogy breaks in two places. First, this is not a storage format you can change at runtime. `G = n_q / H_kv` is frozen into the checkpoint's tensor shapes at pretraining time; there is no re-sharding, no index rebuild, no raising the group size in prod to see what happens. Second — the red box — a compression ratio is a property of the write path, not of the format. The HF reference MLA implementation expands the latent through `kv_b_proj` *before* it calls `cache.update`, caching 20 KiB per layer per token: 17.8x the latent, and five times worse than the GQA-8 it is meant to beat. Measure the cache; never read it off the config.

**What this diagram omits.** All four columns are priced at n_q = 48 on every layer so the comparison is like-for-like. That makes the 1152 KiB MHA baseline a counterfactual, not a shipped number: Laguna really runs 72 query heads on its 36 sliding layers, so a true MHA Laguna is 1584 KiB/token and the shipped saving is 8.25x, not 6x. The UNIFORM note says this in-diagram, but the two figures must not be quoted adjacent to each other without it. No MLA model exists at Laguna's shape. The 54 KiB/token figure drops Kimi Linear's ranks (kv_lora_rank 512, qk_rope_head_dim 64) into Laguna's 48-layer stack. It is arithmetic over two artifacts, not a property of any model that has been trained. The 20 KiB/layer/token gotcha figure is computed from Kimi's own 32 heads, not Laguna's. It does not scale to a 48-layer stack without restating that head count, and the diagram deliberately gives it per layer for that reason. The diagram prices CAPACITY only. It says nothing about whether the bandwidth saving is realised: repeat_kv (modeling_laguna.py:303-312) materialises a G-times copy on any masked layer, and every sliding layer is masked, so the layers with the highest G are structurally guaranteed to take the copying path. That is the curriculum module's section 5.2 and belongs in a separate diagram. Decode arithmetic intensity (AI = 2G/b) is omitted entirely. It is the other half of why H_kv matters, and ASSUMPTIONS.md -> decode-intensity-varies-by-layer records it as derived and never measured; putting a derived number beside measured config reads would give it authority it has not earned. Windowing appears as a single note, not as structure. The growing-term / fixed-term split (12 x 4 KiB per token + 36 x 4 KiB x 512 = 72 MiB) and the fact that the hybrid saves nothing below T = 512 are not drawn. kv_dtype is shown only as b = 2. fp8 or quantised KV storage, which halves bytes and doubles intensity, is not an axis in this diagram. Nothing here is measured on the Z13. Every figure is arithmetic over committed config artifacts read at revision b0a9fd7c850e; the Hardware Validation Gate has not run, so no throughput or residency claim from this machine backs any of it.

*Source: [`docs/diagrams/attention-variants-and-kv-cost.mmd`](diagrams/attention-variants-and-kv-cost.mmd) — 28 grounding references.*

---

## Proteus: the config surface is the experimental surface

```mermaid
flowchart TD
    CFG["ProteusConfig - one typed object loaded from YAML<br/>every field below is an ablation axis<br/>green values are read from laguna-s config.json at b0a9fd7c850e - tag [M]"]

    subgraph SCHED["Layer schedule - the list IS the hypothesis"]
        direction TB
        LIST["layer_types - one string per layer index<br/>read at layer construction, modeling_laguna.py:365<br/>Laguna S ships strict GSSS: 12 full + 36 sliding = 3 to 1"]
        FULL["full_attention<br/>KV residency grows with context"]
        SWA["sliding_attention<br/>sliding_window = 512 tokens"]
        RECUR["linear attention / SSM<br/>no such value exists in LagunaConfig - Proteus-only axis<br/>plug point: Mamba-2 ssm_state, Gated DeltaNet, Samba 1 to 1"]
    end

    subgraph GEOM["Attention geometry - varies BY LAYER TYPE, not by model"]
        direction TB
        QH["num_attention_heads_per_layer<br/>Laguna S: 48 query heads on full, 72 on sliding<br/>top-level num_attention_heads 48 is wrong for 36 of 48 layers"]
        KVH["num_key_value_heads = 8, uniform, no per-layer override<br/>head_dim = 128"]
        GQA["GQA group G = query heads / kv heads<br/>6 on full layers, 9 on sliding<br/>decode arithmetic intensity is 2G / dtype bytes"]
        AGATE["gating = per-head<br/>softplus output gate, one scalar per head<br/>alternative in code: per-element"]
    end

    subgraph POS["Positional scheme - also keyed by layer type"]
        direction TB
        RFULL["full layers: YaRN, rope_theta 500000, factor 128<br/>partial_rotary_factor 0.5, original ctx 8192"]
        RSWA["sliding layers: plain RoPE, rope_theta 10000<br/>partial_rotary_factor 1.0"]
        NOPE["NoPE and the partial-rotary sweep<br/>open axis - nothing in the reference exercises it"]
    end

    subgraph MOEBLK["MoE - per layer, independent of the attention schedule"]
        direction TB
        MTYPE["mlp_layer_types - dense or sparse per layer<br/>Laguna S: layer 0 dense, the other 47 sparse"]
        EXP["num_experts 256, num_experts_per_tok 10<br/>moe_intermediate_size 1024 plus always-on shared expert 1024<br/>moe_routed_scaling_factor 2.5"]
        ROUTER["router: sigmoid, not softmax<br/>e_score_correction_bias with router_aux_loss_coef 0.0<br/>logit softcapping exists in code but ships at 0.0 = OFF"]
    end

    DW["Depth x width<br/>num_hidden_layers x hidden_size = 48 x 3072, intermediate 12288<br/>our ablations run 20M-300M params - tag [A], untested"]

    subgraph BUDGET["Pinned by measurement on gfx1151 - budgets, not axes"]
        direction TB
        KVCOST["KV per token = 2 x 48 x 8 x 128 x 2 bytes = 192.0 KiB exactly<br/>24.0 GiB at 128k context - tag [M]"]
        TIER["fast tier flat at ~200 GB/s out to at least 62 GiB<br/>BIOS UMA carve-out 96 GB - tag [M], one run per arm"]
        BUF["a single tensor of 32 GiB or more hard-hangs at 0 CPU<br/>31 GiB copies cleanly at 199.9 GB/s - tag [M]"]
        ACT["SDPA retains the score matrix: 147.2 bytes per T squared<br/>AOTriton path is 6.6 and stays OFF by default - tag [M]"]
        DIST["torch.distributed is absent from the lab wheel<br/>expert / tensor / pipeline parallel fail at import - tag [M]"]
    end

    CFG --> LIST
    CFG --> QH
    CFG --> KVH
    CFG --> AGATE
    CFG --> MTYPE
    CFG --> DW

    LIST --> FULL
    LIST --> SWA
    LIST --> RECUR

    QH --> GQA
    KVH --> GQA

    FULL -->|"rope_parameters key"| RFULL
    SWA -->|"rope_parameters key"| RSWA

    MTYPE --> EXP
    EXP --> ROUTER

    KVH -.->|"sizes"| KVCOST
    KVCOST -.->|"must fit inside"| TIER
    TIER -.->|"and no single buffer may reach"| BUF
    LIST -.->|"long-context arms are gated by"| ACT
    EXP -.->|"expert count capped single-device by"| DIST
    DW -.->|"param budget capped single-device by"| DIST

    classDef root fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    classDef axis fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    classDef pinned fill:#ffebee,stroke:#c62828,color:#b71c1c

    class CFG root
    class LIST,FULL,SWA,RECUR,QH,KVH,GQA,AGATE,RFULL,RSWA,NOPE,MTYPE,EXP,ROUTER,DW axis
    class KVCOST,TIER,BUF,ACT,DIST pinned
```

This is the config surface as an ablation tree, not a data-flow diagram — nothing here moves a tensor. Each green box is a field you can set, and an arrow means "this field selects that one." The blue root is the single typed config object; the red boxes at the bottom are not axes at all. They are budgets this machine has already measured, and they bound what the green boxes are permitted to say.

The non-obvious thing is how little of the architecture is architecture. The entire SWA/global hybrid question — the one the lab exists to answer — is a list of strings, one per layer, read once at construction (`modeling_laguna.py:365`). Change the list, change the model class. That is why `layer_types` hangs directly off the root instead of sitting inside some attention abstraction: there is no abstraction, and building one would be inventing a requirement.

Two axes are keyed by layer *type* rather than set once for the model, which is easy to miss from `config.json` alone. Query-head count is per layer (48 on full, 72 on sliding — the top-level `num_attention_heads: 48` is wrong for 36 of 48 layers), and the positional scheme is a different RoPE per type: YaRN at θ=500000 over half the head dims on full layers, plain RoPE at θ=10000 over all of them on sliding. You cannot simply widen the sliding windows to test long context; those layers were never trained with positional encoding that reaches past 512.

Where the systems analogy breaks: the red boxes look like a capacity plan, and the KV arithmetic genuinely is one — 192.0 KiB/token, 24.0 GiB at 128k, against a fast tier flat to at least 62 GiB. But a sliding layer does not *evict* the tokens outside its window. They are architecturally unreadable, so discarding them is lossless. No hit rate, no miss path, no tier to promote from.

**What this diagram omits.** The tree shows the config surface of the REFERENCE model (Laguna S 2.1) plus the axes Proteus intends to add. Proteus itself has no config code yet — `packages/proteus/src/proteus/__init__.py` is the only source file and `configs/` holds only a README. Every green box is a plan, not a shipped field. The 'linear attention / SSM' box has no config vocabulary anywhere. Laguna's `layer_types` validator accepts only `full_attention` and `sliding_attention`; the Mamba-2 / Gated DeltaNet / Samba references are where the mechanism would be read from, not a value that exists. The `NoPE and partial-rotary sweep` node deliberately has no incoming edge — no shipped config sets it. That is information, not a layout accident. The diagram omits every axis that is not per-layer or not experimental: vocab_size, tokenizer, rms_norm_eps, initializer_range, attention_bias, dropout, QK-norm (present in code at modeling_laguna.py:368 but not a planned axis), and the whole `base_model_tp_plan` / `ep_plan` surface (dead on this hardware — see the DIST box). `GQA group G = 6 / 9` is a measured config read; the claim that decode arithmetic intensity is 2G/dtype_bytes is DERIVED, not measured (ASSUMPTIONS row `decode-intensity-varies-by-layer`). The diagram states the formula without a measurement tag for that reason. The `at least 62 GiB` fast tier is a single run per arm — an anecdote by this lab's own >=3-seed standard. The upper edge was never found; the sweep hit the 32 GiB tensor fault, not a bandwidth knee. The red budgets are properties of gfx1151 under the current wheel (`torch 2.12.0a0+rocm7.13.0a20260313`). They are re-measured after any ROCm/PyTorch change; an upgrade invalidates the bottom third of this diagram. hipBLASLt configuration is not drawn. It is a per-run recorded control (ADR: hipblaslt-is-a-numerics-control), not a model axis, but it confounds any long-context result taken without it. MoE parameter counting is not shown. At 20M-300M ablation scale, `num_experts: 256` is not reachable; the real ablatable range is far smaller and nobody has costed it yet. I could not run mermaid-cli in this session (no shell tool available). Syntax was hardened by inspection: all labels quoted, no `<`/`>` outside `<br/>`, no `#` or `;` inside labels, dotted-edge labels in the `-.->|"..."|` form.

*Source: [`docs/diagrams/proteus-config-surface.mmd`](diagrams/proteus-config-surface.mmd) — 35 grounding references.*

---

# System layouts

## Four systems, and the boundary that is enforced

```mermaid
flowchart TB
  subgraph chiron["Chiron — the lab: repo, docs, curriculum, governance. Owns no runtime code."]
    direction TB

    governance["docs/adr/ — decision records, frozen once Accepted<br/>docs/adr/README.md is the register: slug, status, date, body SHA-256<br/>tests/test_adr_immutability.py recomputes the hashes and fails on mismatch"]

    ledgers["ASSUMPTIONS.md — assumption, status, evidence, date<br/>notebook/ — pre-registered hypothesis cards and results<br/>Binding number for all KV work: fast tier ≥62 GiB at ~200 GB/s, measured 2026-07-26"]

    guard["BOUNDARY ENFORCEMENT — a mechanism, not discipline<br/>packages/mnemosyne/pyproject.toml declares torch only — no proteus, no themis,<br/>so an accidental import fails at dependency resolution rather than surviving review<br/>tests/test_package_boundaries.py AST-walks packages/mnemosyne/src and fails,<br/>naming the offending file, on any import of proteus or themis<br/>Proved red-then-green 2026-07-26; separability itself is still untested until<br/>the clean-venv wheel acceptance test runs at the mnemosyne-core milestone"]

    subgraph themis["Themis — packages/themis — the ablation rig"]
      direction TB
      themis_rig["Owns: pre-registration, matched param and token budgets,<br/>seed management ≥3, run execution, aggregation, confidence intervals"]
      argus["argus/ — a module, not a package<br/>telemetry: JSONL loss, LR, throughput, cache-stats<br/>Reserved name; earns a package only if it outgrows the module"]
      themis_data["data/ — a module, not a package<br/>seeded resumable loaders plus targeted recall and retrieval probes<br/>Deliberately unnamed: no interface or lifecycle of its own yet"]
    end

    subgraph proteus["Proteus — packages/proteus — the model architecture"]
      direction TB
      proteus_cfg["Owns: decoder, attention variants, MoE gating, positional schemes.<br/>Every ablation axis is a config field; the config surface IS the experimental surface.<br/>Reference shape read from laguna-s config.json: 48 layers as<br/>12 full_attention plus 36 sliding_attention, sliding_window 512, 256 experts with 10 active"]
    end

    subgraph mnemosyne["Mnemosyne — packages/mnemosyne — the memory subsystem, and THE research contribution"]
      direction TB
      mnemo_cache["Owns: KV cache, tiering, prefix reuse, compression, attribution telemetry.<br/>Sizing target from the reference config: 2·48·8·128·2 B = 192.0 KiB per token,<br/>so 24.0 GiB at 128k context against the ≥62 GiB fast tier"]
      plugs["Public interface: three plug points —<br/>write-time admission, deferred eviction, read-time selection.<br/>One score-then-subset hook cannot host the existing literature"]
      lethe["Lethe — RESERVED, no code, no directory<br/>the eviction-policy layer; the name is spent only once<br/>there is more than one policy to hold"]
    end
  end

  torch["torch — the only dependency Mnemosyne is permitted<br/>2.12.0a0+rocm7.13.0a20260313 on gfx1151, native Windows<br/>Single device: torch.distributed is absent from this build, so every parallelism path fails at import"]

  themis ==>|"depends on"| proteus
  themis ==> mnemosyne
  proteus ==>|"requests cache through Mnemosyne's public interface"| mnemosyne
  themis --> torch
  proteus --> torch
  mnemosyne --> torch
  mnemosyne -.->|"THE EDGE THAT MUST NOT EXIST<br/>proteus and themis are invisible to mnemosyne by construction"| proteus
  guard -.->|"enforces"| mnemosyne

  classDef contribution fill:#eef6ff,stroke:#0b63c5,stroke-width:2px,color:#0b2d4e
  classDef reserved fill:#f4f4f4,stroke:#8a8a8a,stroke-width:1px,color:#4a4a4a
  classDef control fill:#fff4f4,stroke:#cc0000,stroke-width:2px,color:#6b0000
  classDef record fill:#f7f4ff,stroke:#6b4fbb,stroke-width:1px,color:#31215e

  class mnemo_cache,plugs contribution
  class lethe,argus reserved
  class guard control
  class governance,ledgers record

  linkStyle 6 stroke:#cc0000,stroke-width:4px
  linkStyle 7 stroke:#cc0000,stroke-width:2px
```

The four boxes are the whole lab. Chiron is the repository and its governance — records, registers, and the tests that keep them honest — and owns no runtime code. The three inner boxes are separately distributable packages in one uv workspace, and the arrows between them are their *declared* dependencies, drawn top-to-bottom so that "depends on" points down.

The non-obvious thing is the red dashed arrow, which is drawn precisely because it does not exist. Mnemosyne's `pyproject.toml` lists `torch` and nothing else. That omission is not tidiness; it is the research claim rendered as a build constraint. A memory subsystem that only works against our own decoder is an implementation detail rather than a contribution, so separability has to be mechanically true instead of asserted. Two artifacts hold it: the missing dependency, which makes an accidental `import proteus` fail at resolution, and `tests/test_package_boundaries.py`, which AST-walks Mnemosyne's source and fails *naming the offending file*, so the error is legible rather than cryptic. The guard itself has been exercised red-then-green; separability has not — that waits on a clean-venv wheel install at the `mnemosyne-core` milestone.

Where the systems reading breaks: this looks like a layering diagram, where the bottom is the most stable substrate and the arrows describe blast radius. It is not. Mnemosyne sits lowest because of what it must be detachable *from*, not because it changes least — it is the layer under the heaviest development. And `torch` here is not a swappable platform: it is one pinned ROCm nightly on one iGPU, with `torch.distributed` absent from the build entirely, so there is no horizontal escape hatch under the stack. The capacity and bandwidth figures in the boxes are single-run instrument characterisation, not results.

**What this diagram omits.** The arrows are DECLARED dependencies from pyproject.toml, not a runtime call graph. There is no call graph yet: `packages/**/src/**/*.py` currently contains five `__init__.py` files and nothing else, so no code exists for any arrow to describe. The three plug points on Mnemosyne (write-time admission / deferred eviction / read-time selection) come from `docs/adr/attribution-instrument-over-eviction-policy.md`, which is Status: **Proposed**, awaiting founder review. That node is drawn as if settled and is not. The diagram shows the boundary GUARD, not proven separability. `ASSUMPTIONS.md` row `mnemosyne-separable` reads "guard proven; separability itself untested" — the clean-venv wheel acceptance test has not run. Every number in a label is single-run instrument characterisation (>=62 GiB, ~200 GB/s), which the house standard calls an anecdote. The Hardware Validation Gate has not run, so nothing here counts as a research result. The Laguna numbers in the Proteus box describe the REFERENCE model under study, read from `research/reference/models/laguna-s/config.json`. They are not Proteus's own implemented shape — Proteus has no model code. The 192.0 KiB/token figure is exact for a full unwindowed cache. Actual residency is far lower because 36 of 48 layers are windowed at 512 — the diagram states the ceiling, not the working set. `torch.distributed` being absent is a property of the currently pinned wheel (`2.12.0a0+rocm7.13.0a20260313`), not of the hardware. A different wheel or rented hardware changes it. Chiron's other contents are omitted: `curriculum/`, `research/` (reference clones, memory track, notes), `configs/`, `BACKLOG.md`, `BLOCKERS.md`, `CHANGELOG.md`, and root `tests/` beyond the two guard tests named. Only two of the four ADRs' enforcement paths are shown (the hash register and the boundary lint). The `Proposed`/`Accepted`/`Superseded` status ladder itself is not drawn. `>=62 GiB` is a floor, not an edge — the bandwidth sweep never found where degradation begins; it hit the >=32 GiB single-tensor hang first. The diagram states it as a budget, which is how the notebook entry recommends using it, but it is not a measured tier boundary.

*Source: [`docs/diagrams/four-systems-and-boundaries.mmd`](diagrams/four-systems-and-boundaries.mmd) — 24 grounding references.*

---

## One ablation run, end to end

```mermaid
sequenceDiagram
    autonumber
    actor Res as Researcher
    participant Card as Notebook record<br/>frozen G2 card
    participant Git as Git history<br/>commit + semver tag
    participant Themis as Themis<br/>ablation rig
    participant Proteus as Proteus<br/>model under test
    participant Mnem as Mnemosyne<br/>KV cache + plug points
    participant Argus as Argus<br/>telemetry JSONL
    participant Ledger as ASSUMPTIONS.md<br/>register

    rect rgb(238, 242, 248)
    Note over Res,Git: PRE-REGISTRATION GATE - completes before any GPU work starts
    Res->>Card: write the G2 card verbatim - HYPOTHESIS, FOR, BECAUSE, MEASURED BY, SUCCESS, KILL, COST, RISKIEST
    Res->>Card: freeze the design - arms, matched param and token budgets, seed list, named confounder
    Card->>Git: commit the card on a clean tree and tag it
    Git-->>Res: the card is now a record. SUCCESS and KILL can no longer move
    end

    Res->>Themis: run the pre-registered arm set

    Themis->>Themis: resolve configs/swa-4to1-ctx32k.yaml into one typed config object
    Note right of Themis: every ablation axis is a config field.<br/>Reference model reads 48 layer_types entries,<br/>sliding_window 512, 8 KV heads, head_dim 128
    Themis->>Argus: stamp the environment fingerprint - torch wheel, HIPBLASLT_TENSILE_LIBPATH set, AOTriton flag off

    loop once per seed - 3 or more, never 1
        Themis->>Proteus: build the model at this seed with param count matched across arms
        Proteus->>Mnem: request cache for 48 layers of 8 KV heads by 128 dim
        Mnem-->>Proteus: cache handle - 192.0 KiB per token, so 24.0 GiB at 128k context
        Note over Mnem: fits the measured 62 GiB fast tier at ~200 GB/s.<br/>Keep every single tensor under 32 GiB -<br/>larger ones hang silently at 0 CPU

        loop once per step
            Proteus->>Mnem: write K and V, then read through admission / eviction / read-selection
            Mnem->>Mnem: replay the step against a full-cache oracle and diff per-token KL
            Mnem-->>Argus: attribution record - which plug point dropped which token
            Proteus-->>Argus: loss, LR, throughput, cache stats
        end
        Argus->>Argus: buffer metrics as device tensors, flush on an interval - reading one stalls the GPU

        Themis->>Themis: checkpoint into a temp dir then rename, verify the reload is bit-exact
        Argus-->>Themis: one JSONL run file for this seed
    end

    Themis->>Themis: aggregate across seeds - mean, confidence interval, seed-to-seed null distribution
    Themis->>Card: read SUCCESS and KILL for the first time
    Note over Themis,Card: the thresholds are opened only now.<br/>Moving one here is a change of standard and must be declared as one

    alt primary metric clears SUCCESS
        Themis-->>Res: adopt the arm
    else primary metric hits KILL
        Themis-->>Res: drop the axis and write it up with equal care
    else lands between the two
        Themis-->>Res: partial - the mechanism stays open
    end

    Res->>Card: append the Results section once, then it freezes too
    Res->>Ledger: update the assumption row - status, evidence tag, date
```

One ablation run, read top to bottom. The shaded block is the reason this is a sequence diagram and not a flowchart: the hypothesis card — HYPOTHESIS through RISKIEST — is committed and tagged at steps 3–4, and Themis does not open SUCCESS or KILL until step 19, after aggregation. Everything in between is executed by machinery that cannot see the thresholds it will be judged against. Reverse that order and the run still produces numbers; it stops producing evidence.

The three-way `alt` at the bottom is load-bearing. A KILL is a completed experiment written up with the same care as a SUCCESS, and the middle branch — landing between thresholds — is pre-declared as partial rather than argued into a win afterwards.

The non-obvious step is 12. Mnemosyne replays each probe step against a full-cache oracle: attribution requires running the expensive thing eviction exists to avoid. That is affordable at 20M–300M parameters and unaffordable at 70B, which is why small scale is the enabling condition here rather than a compromise.

Where the systems analogy breaks, twice. Step 15 looks like a write-ahead log with group commit — buffer records, flush on an interval. It is not: the buffer is volatile, and the batching does not amortize I/O. Metrics are held as unevaluated device tensors because reading one forces a host-device sync that stalls the GPU. Observability is charged directly against training throughput, which is not true of any logging system you have run in production. Step 16 breaks differently: the checkpoint has no journal and no incremental delta. Every save is a full rewrite committed by directory rename, so atomicity is rename-granularity — a torn save loses the whole checkpoint, not a tail.

**What this diagram omits.** This is design, not an observed trace. Themis, Proteus, Mnemosyne and Argus are scaffolded packages at version 0.1.0 containing only `__init__.py` docstrings — no run has ever taken this path. The ADR that specifies the three plug points and the full-cache oracle harness (`attribution-instrument-over-eviction-policy`) is still `Proposed`, awaiting founder review. Steps 11–13 draw a decision that has not been accepted. One arm is drawn. A real ablation runs at least two matched arms (e.g. `proteus-swa-4to1` vs `proteus-dense`); the arm loop is collapsed into the single "run the arm set" message. The oracle diff is drawn as if it runs every step. Per the ADR it runs on probes, not on every training step — the diagram trades that precision for a readable repeating unit. The config values shown (48 layers, 8 KV heads, head_dim 128, sliding_window 512) and the derived 192.0 KiB/token / 24.0 GiB at 128k are the *reference* model's, read from Laguna S 2.1's config.json. Our own ablations run at 20M–300M params, so those are the numbers being studied, not the numbers our arms will cost. Residency in practice is far below 24.0 GiB because 36 of 48 layers are windowed at 512 tokens. The diagram states the full-residency upper bound because that is what a capacity budget must clear. The Hardware Validation Gate is not drawn and has not passed. bf16 numerics, determinism and checkpoint integrity remain unproven, so nothing produced by this lifecycle counts as evidence yet. Failure paths are omitted entirely: OOM, preemption, resume-from-checkpoint, and the silent >=32 GiB hang at 0 CPU that would stall this sequence without raising an error. No distributed path is shown, and that is not a simplification — `torch.distributed` is absent from the lab wheel, so every parallelism path fails at import. The `>=3 seeds` loop is drawn as sequential. Whether seeds run sequentially or share a process is unspecified in any committed artifact.

*Source: [`docs/diagrams/experiment-lifecycle.mmd`](diagrams/experiment-lifecycle.mmd) — 39 grounding references.*

---

## Mnemosyne's contract with the outside world

```mermaid
flowchart TB

  write["PROTEUS produces k_i, v_i<br/>8 KV heads x head_dim 128 x 2 B = 192 KiB per token, exact<br/>48 layers: 12 full_attention, 36 sliding_attention at window 512"]
  query["PROTEUS emits q_t - one query per decode step"]
  read["PROTEUS computes o_t = sum of a_i v_i<br/>over whatever Mnemosyne returned"]

  onehook["The hook that does NOT work:<br/>score(keys, values, attention) -> subset"]

  subgraph mnem["Mnemosyne - the public contract. Never imports proteus or themis."]
    direction TB
    admit["WRITE-TIME ADMISSION<br/>sees k_i, v_i only. O(d), no attention matrix<br/>hosts L2-norm, KeyDiff, EpiKV"]
    stage["staging buffer - hold k steps, then flush<br/>k steps of extra residency buys k steps of context"]
    defer["DEFERRED EVICTION<br/>sees prompt-side or k-step-lagged attention<br/>hosts H2O, SnapKV, PyramidKV, ChunkKV, KVpop"]
    resident["resident set - budget B, per layer and per head<br/>12 global layers grow with context, 36 stay at 512<br/>budget against the 62 GiB-or-more fast tier at ~200 GB/s [M];<br/>keep any single tensor under 32 GiB [M]"]
    select["READ-TIME SELECTION<br/>sees the actual q_t<br/>hosts Quest, SparQ, RocketKV stage 2"]
    emit["TELEMETRY, mandatory - two scalars per step, per layer, per head.<br/>Error is at most (1 - A) x max distance from a dropped v_i to the retained output."]
  end

  gone["dropped - no backing store, no miss path, no hit-rate metric.<br/>The only refetch is re-prefilling the whole prefix."]

  fastkv["OUT OF CONTRACT - FastKV / Token-Selective Propagation.<br/>A token not propagated past layer L never has KV computed for L+1..48:<br/>not evicted, never produced. It edits Proteus's forward pass."]

  massA["A - retained attention mass<br/>every score-based policy optimises this"]
  dist["dropped-value distance<br/>no published policy reports this one"]
  ratios["compression is TWO numbers, always -<br/>bytes held vs bytes read per step<br/>RocketKV: 400x against 32.6%"]

  subgraph inv["Invariants - every plug point above is held to all four"]
    direction LR
    i1["eviction is a bet on correctness,<br/>not on latency - it is irreversible"]
    i2["dropping reweights the survivors -<br/>the softmax denominator shrinks"]
    i3["attention sinks stay pinned -<br/>load-bearing, not hot"]
    i4["every policy stays bypassable -<br/>the full-cache oracle is the instrument"]
  end

  write --> admit
  admit -->|"admit"| stage
  admit -->|"reject"| gone
  stage --> defer
  defer -->|"survivors"| resident
  defer -->|"evict"| gone
  query --> select
  resident -->|"candidates"| select
  select -->|"subset read this step, cache unchanged"| read

  onehook -.->|"KVpop needs a staging buffer and a flush trigger"| defer
  onehook -.->|"RocketKV stage 2 needs q_t, which does not exist yet"| select
  onehook -.->|"FastKV cannot be hosted at any depth"| fastkv

  admit -.-> emit
  defer -.-> emit
  select -.-> emit
  emit --> massA
  emit --> dist
  emit --> ratios

  classDef proteus fill:#e8eeff,stroke:#3a4a80,color:#111
  classDef outside fill:#ffecec,stroke:#8a3a3a,color:#111
  class write,query,read proteus
  class gone,fastkv,onehook outside
```

This is Mnemosyne's public contract, not its implementation. Read the blue nodes as Proteus: it produces `k_i` and `v_i` on the way in, emits `q_t` once per decode step, and consumes a subset on the way out. Everything between is the surface a memory policy plugs into. The three plug points differ in exactly one respect — when they run relative to the query — and that ordering is the entire design. Write-time admission sees only the key and value of the token being written: O(d) work, no attention matrix in scope. Deferred eviction runs after a bounded staging delay and sees prompt-side or k-step-lagged attention. Read-time selection runs per step and is the only place the actual `q_t` exists.

The non-obvious thing is why the obvious hook fails. `score(keys, values, attention) -> subset` is a *ranking* interface, and this is not a ranking problem — nine of the ten canonical policies commit to a retained subset before the query that will read it exists. The three dashed edges are the three ways that bites: KVpop needs a buffer to defer in, RocketKV's second stage needs a query nobody has emitted yet, and FastKV cannot be hosted at any depth, because it discards tokens inside the forward pass so their KV is never produced rather than evicted.

Where the caching analogy breaks: eviction here is a bet on correctness, not latency. No backing store, no miss path, and — the part that matters operationally — no hit-rate metric. A wrong eviction returns a fluent, confident, wrong answer with no error signal anywhere in the stack. That is why telemetry sits inside the contract rather than beside it.

**What this diagram omits.** The contract is Proposed, not Accepted. Both `research/synthesis.md` (status: proposed — awaiting founder review) and ADR `attribution-instrument-over-eviction-policy` (Status: Proposed) still await founder review. The diagram draws a proposal, not a decision in force. The four invariants are my synthesis, not a specialist's list. i1–i3 are read off §2's 'four breaks' in `kv-compression-and-eviction.md`; i4 comes from the ADR's oracle-diff requirement. The note never labels anything an invariant and nobody has signed off on this as the complete set. Gap flagged to memory-systems-researcher. No API shape anywhere. The source settles *when* each plug point runs and *what it can see*; it says nothing about signatures, who allocates and owns the cache tensors, the flush-trigger semantics of the staging buffer, or how read-time selection composes with a paged/block layout. The diagram shows a contract with no types in it. Gap flagged to ml-architect. 'budget B, per layer and per head' extrapolates. The note logs telemetry per layer and per head, but whether the *budget* should be allocated per layer, per head, or uniformly is live dispute §5.3 (PyramidKV vs Ada-KV vs LAVa). The diagram shows the axis exists; it does not claim a rule. The ≥62 GiB fast tier at ~200 GB/s is one run per arm — an anecdote by the house ≥3-seed standard — and its upper edge is unmeasured (the sweep hit the 32 GiB tensor fault, not a bandwidth knee). The 32 GiB fault rests on two observations with an untested mechanism. Nothing measured on this machine is admissible yet: `bf16-numerics-unproven` is untested and the Hardware Validation Gate has not run. Every [M] on the diagram is provisional in that sense. 192 KiB/token is exact for Laguna S 2.1 and for an all-global cache. Real residency is roughly 4x lower because 36/48 layers window at 512. The diagram states both facts and computes neither into a residency figure. The diagram is decode-shaped: one layer, one head, one step. It omits prefix reuse and cross-request sharing, quantization, offload to a slow tier, and the batch dimension — all of which the serving-hierarchy note treats and all of which will need their own diagram. FastKV is drawn as a single out-of-contract example. It stands in for a class (anything that alters the forward pass, including trained sparsity like NSA/DSA); the class boundary has not been enumerated.

*Source: [`docs/diagrams/mnemosyne-cache-interface.mmd`](diagrams/mnemosyne-cache-interface.mmd) — 27 grounding references.*

---

## The memory hierarchy, as measured here

```mermaid
flowchart TB

  kv_budget["What has to fit: the Laguna S 2.1 KV cache<br/>2 x 48 layers x 8 KV heads x 128 head_dim x 2 B = 192.0 KiB per token<br/>24.0 GiB at 128k context, computed from config.json, exact"]

  subgraph measured_here["THIS MACHINE, MEASURED 2026-07-26 - Z13, gfx1151, one unified 128 GB pool, single run per arm"]
    carveout_knob["BIOS UMA FB Size - THE KNOB<br/>one boot-time field moves the fast/slow boundary<br/>no discrete GPU exposes this control at all"]
    carveout_default["carve-out at the 16 GiB default, the before arm<br/>fast tier ends at a 30 GiB footprint<br/>194.9 GB/s at 30 GiB, 61.3 GB/s at 32 GiB, 114.1 GB/s at 60 GiB<br/>reported pool 82.99 GiB, Windows sees 111.6 GB"]
    fast_tier["FAST TIER, carve-out at 96 GB - the planning number<br/>at least 62 GiB, flat at 203-205 GB/s, no knee anywhere swept<br/>a floor, not an edge: the sweep hit the wall below before it found one"]
    tensor_wall["HARD WALL INSIDE THE FAST TIER - any single tensor at or above 32 GiB<br/>31 GiB copies clean at 199.9 GB/s<br/>32 GiB hard-hangs at 0 CPU, silently, no error, force-killed at 11 min<br/>36 GiB raises hipErrorLaunchFailure"]
    slower_region["SLOWER REGION past the fast tier<br/>reported pool 107.87 GiB, and at least 74.40 GiB written, read back, released<br/>its bandwidth at the 96 GB carve-out is UNMEASURED"]
    oversubscribed["the pool figure is not a ceiling<br/>allocation-only probes returned at least 100 GiB<br/>the driver oversubscribes into system RAM"]
    host_side["HOST - Windows visible RAM 31.6 GB<br/>the same physical DRAM as the fast tier, with no bus to cross<br/>bandwidth assumed comparable, never measured"]
  end

  subgraph assumed_by_literature["THE MACHINE THE LITERATURE ASSUMES - cited, the ladder the offload and CXL designs are built around"]
    gpu_hbm["GPU HBM<br/>TB/s, tens of GiB<br/>size fixed at purchase"]
    pcie_link["PCIe<br/>tens of GB/s"]
    host_dram["Host DRAM<br/>about 100 GB/s"]
  end

  ratio_contrast["THE RATIO IS THE WHOLE ARGUMENT<br/>theirs: 10-50x, set by a bus, not variable<br/>ours: 2-3x, set by a BIOS field, swept in one reboot"]
  research_consequence["Retention beats eviction as that ratio falls, because a refetch costs<br/>bytes over the slow path and a recompute does not.<br/>So a class of published KV-tiering guidance may be a claim about interconnects<br/>rather than about language models - testable here, and on no card we own."]

  kv_budget -->|"budget against the fast tier, never against the pool"| fast_tier
  carveout_knob -->|"one reboot"| carveout_default
  carveout_knob -->|"one reboot, kept - see ADR bios-uma-carveout-at-96gb"| fast_tier
  fast_tier -.->|"applies to any single buffer, whatever the total footprint"| tensor_wall
  fast_tier -->|"past the measured floor"| slower_region
  slower_region -.-> oversubscribed
  slower_region -->|"same pool, shared with the OS"| host_side
  gpu_hbm -->|"10-50x drop, across the bus"| pcie_link
  pcie_link --> host_dram
  slower_region -.->|"measured"| ratio_contrast
  pcie_link -.->|"assumed"| ratio_contrast
  ratio_contrast --> research_consequence

  classDef wall fill:#ffe3e3,stroke:#c92a2a,stroke-width:3px,color:#000
  classDef knob fill:#e7f0ff,stroke:#1c62d1,stroke-width:3px,color:#000
  classDef unmeasured fill:#f1f3f5,stroke:#868e96,color:#000
  classDef anchor fill:#fff9db,stroke:#e8a90c,color:#000

  class tensor_wall wall
  class carveout_knob knob
  class slower_region,oversubscribed,host_side unmeasured
  class kv_budget,ratio_contrast,research_consequence anchor
```

The left ladder is this machine as measured on 2026-07-26. The right one is the machine nearly every KV-offload and CXL-tiering paper is written for. They are the same picture drawn twice, and the gap between them is this lab's main structural advantage.

Three numbers carry the diagram. The fast tier is at least 62 GiB at a flat 203–205 GB/s, and that — not the 107.87 GiB the driver reports, not the ≥74.40 GiB that has been written and read back — is what every long-context experiment is sized against. Laguna S 2.1 costs 192.0 KiB of KV per token exactly, so a 128k context is 24.0 GiB and fits with room. Inside the fast tier sits a hard wall: any *single* tensor at or above 32 GiB. A 31 GiB buffer copies clean at 199.9 GB/s; a 32 GiB buffer hangs at zero CPU with no error. That wall is orthogonal to capacity — it does not care how much is free — and it stalls a run rather than crashing it.

The non-obvious element is the blue node. The fast/slow boundary here is a BIOS field, not a bus. Moving it from the 16 GiB default to 96 GB moved the boundary from 30 GiB to ≥62 GiB in one reboot. No discrete GPU can vary that ratio at all, which turns a fixed premise of the literature into a swept variable.

Where the storage-hierarchy analogy breaks: there is no change of medium and no miss path. Every box on the left is the same DRAM. Nothing is promoted, nothing demoted, nothing faults in — and "offload to host DRAM," the move the offload literature is built on, buys no capacity here, because the host's share shrank to 31.6 GB out of that same pool. These tiers are an allocation policy wearing a hierarchy's clothes.

**What this diagram omits.** Every measured number on the left ladder is a single run per arm — an anecdote by this repo's own standard (>=3 seeds, CIs). The effect sizes are large and the boundaries sharp across adjacent points, which is why they are reported, but nothing here is a research result. The >=62 GiB fast tier is a floor, not a measured edge. The sweep never found a bandwidth knee — it hit the >=32 GiB single-tensor fault first. Whether the fast tier ends at 62, 80, or 96 GiB is unknown. The 'slower region' box is drawn from the 16 GiB-carve-out arm, where bandwidth past 30 GiB fell to 61–114 GB/s. At the 96 GB carve-out currently in force, the region beyond 62 GiB was never swept, so its bandwidth is unmeasured. The diagram says so on the node; do not read the box as a measurement. Host-side bandwidth is assumed comparable to device-to-device, not measured. The cheapest test (host-to-device copy plus CPU memcpy on the same box, against the ~200 GB/s figure) has not been run. The right-hand ladder is the literature's design premise, cited, not benchmarked by us. TB/s, ~100 GB/s and 'tens of GB/s' are order-of-magnitude framings from the survey notes, not vendor specs for a named part. 24.0 GiB at 128k is the worst case with all 48 layers holding full context. In practice 36 of 48 Laguna layers are windowed at 512 tokens, so real residency is far lower — a window effect the diagram does not draw. The 32 GiB wall's mechanism is unexplained. A 32-bit overflow in the copy path is a medium-confidence hypothesis, not a finding; the discriminating test (32 GiB as fp32) has not been run. The diagram states the symptoms only. No temperature, power profile, or repeat-count control is represented. Thermal throttling was checked and cleared as a confounder for the boundary move, but only by the argument that small-footprint bandwidth was unchanged.

*Source: [`docs/diagrams/memory-hierarchy-measured.mmd`](diagrams/memory-hierarchy-measured.mmd) — 22 grounding references.*

---

## PagedAttention: a page table, until it isn't

```mermaid
flowchart TB

  subgraph REQS["Block table, scheduler side - one ordered list per request, per KV cache group"]
    direction LR
    AL0["A logical 0<br/>tokens 0-15"]
    AL1["A logical 1<br/>tokens 16-31"]
    AL2["A logical 2<br/>tail, partial, never hashed"]
    BL0["B logical 0<br/>same 16 tokens"]
    BL1["B logical 1<br/>same 16 tokens"]
    BL2["B logical 2<br/>tail, partial, never hashed"]
  end

  subgraph POOL["One preallocated KV pool - physical frames, DEFAULT_BLOCK_SIZE = 16 tokens, cache.py:47"]
    direction LR
    F12["frame 12<br/>ref_cnt = 2<br/>hashed, pinned"]
    F19["frame 19<br/>ref_cnt = 2<br/>hashed, pinned"]
    F31["frame 31<br/>ref_cnt = 1"]
    F44["frame 44<br/>ref_cnt = 1"]
    SIZING["Pool sizing on the lab machine<br/>Laguna S 2.1 = 192.0 KiB per token, so 24.0 GiB at 128k ctx<br/>fast tier measured at >= 62 GiB sustaining ~200 GB/s<br/>but any single tensor >= 32 GiB hangs, so the pool must be chunked"]
  end

  subgraph LOOKUP["Prefix hit path - runs before allocation, never after"]
    direction TB
    MATCH["find_longest_cache_hit<br/>walk block hashes from token 0, break on the FIRST miss,<br/>floor to alignment - single_type_kv_cache_manager.py:708"]
    HASH["cached_block_hash_to_block probe<br/>key = hash of parent_block_hash + token_ids + extra_keys<br/>kv_cache_utils.py:596, block_pool.py:198"]
    TOUCH["touch - ref_cnt += 1 and O(1) unlink from the free queue<br/>THIS is what makes a matched block un-evictable<br/>block_pool.py:702"]
  end

  subgraph FREEQ["FreeKVCacheBlockQueue - the free list and the LRU victim cache are the same list, kv_cache_utils.py:184"]
    direction LR
    FFRONT["FRONT - reclaimed first<br/>unhashed blocks, prepend_n<br/>block_pool.py:741"]
    FMID["frame 88 - ref_cnt 0, KV contents and hash entry INTACT<br/>still matchable, still resurrectable by touch"]
    FBACK["BACK - survives longest<br/>hashed blocks, append_n<br/>block_pool.py:742"]
  end

  FREEB["free_blocks on reversed request blocks - ref_cnt -= 1<br/>freeing is NOT evicting; the hash entry stays live<br/>single_type_kv_cache_manager.py:503, block_pool.py:719"]
  ALLOC["get_new_blocks - popleft_n, then _maybe_evict_cached_block<br/>drops the stale hash entry, sets ref_cnt = 1, no zeroing<br/>eviction happens HERE, at reallocation - block_pool.py:647, 679"]
  KWALK["Worker side - the same table flattened to a dense int32 matrix<br/>of max_num_reqs by max_num_blocks_per_req, copied to GPU each step,<br/>then walked inside the attention kernel, one load per KV tile.<br/>No MMU, no TLB - worker/block_table.py:81, triton_unified_attention.py:424"]

  AL0 --> F12
  AL1 --> F19
  AL2 --> F31
  BL0 -->|"prefix hit"| F12
  BL1 -->|"prefix hit"| F19
  BL2 --> F44

  MATCH --> HASH
  HASH -->|"hit"| TOUCH
  TOUCH -->|"ref_cnt 1 to 2"| F12
  TOUCH -->|"ref_cnt 1 to 2"| F19
  TOUCH -.->|"unlink on hit"| FMID
  HASH -->|"miss"| ALLOC

  FREEB --> FFRONT
  FREEB --> FBACK
  FFRONT --> FMID
  FMID --> FBACK
  FFRONT -->|"popleft_n"| ALLOC
  ALLOC --> F31
  ALLOC --> F44

  POOL --> KWALK

  subgraph BREAKS["Four places the virtual-memory analogy breaks"]
    direction TB
    BRK_FAULT["No fault path.<br/>allocate_slots returns None when the pool runs dry<br/>kv_cache_manager.py:484 and 523. Nothing is demand-paged<br/>and there is no lower tier to fault in from."]
    BRK_GRAIN["Eviction granularity is the REQUEST, not the page.<br/>The scheduler preempts a whole sequence, frees every block it holds<br/>and resets num_computed_tokens to 0<br/>scheduler.py:566-613 and 1212-1216."]
    BRK_KEY["The key is a prefix CHAIN, not a content hash.<br/>hash_block_tokens folds in parent_block_hash, so the same 16 tokens<br/>at a different offset are a different key, and identical blocks<br/>are deliberately never de-duplicated - block_pool.py:47-51."]
    BRK_LOSS["Discarding is always legal.<br/>KV is recomputable from the token ids, so a wrong eviction costs<br/>one prefill and never costs correctness. No storage tier you have<br/>operated is permitted to make that trade."]
  end

  BRK_FAULT -.-> ALLOC
  BRK_GRAIN -.-> FREEB
  BRK_KEY -.-> HASH
  BRK_LOSS -.-> ALLOC

  classDef analogybreak fill:#fff4e6,stroke:#d97706,stroke-width:2px,color:#111111;
  classDef pinnedframe fill:#e6f4ea,stroke:#137333,color:#111111;
  classDef sizingnote fill:#f1f3f4,stroke:#5f6368,color:#111111;
  class BRK_FAULT,BRK_GRAIN,BRK_KEY,BRK_LOSS analogybreak;
  class F12,F19 pinnedframe;
  class SIZING,KWALK sizingnote;
```

You are looking at a page table with the hardware taken out. `req_to_blocks` is one plain Python list per request per KV cache group; index `i` is logical KV block `i`, and the `block_id` stored there is the physical frame in a single preallocated pool of 16-token blocks. Requests A and B share a system prompt, so their first two logical slots resolve to the same two frames. That sharing happens once, at lookup time, before allocation — never retroactively. `touch` bumps `ref_cnt` and unlinks the frame from the free queue in O(1), and that refcount is the whole pinning mechanism: a frame with `ref_cnt > 0` cannot be handed out.

The non-obvious thing is the free list, because it is not a free list. `FreeKVCacheBlockQueue` holds `ref_cnt == 0` blocks with their KV contents *and* their hash entries intact, so a later request can resurrect one on a prefix hit. Freeing is not evicting; eviction happens lazily inside `get_new_blocks`, at the moment a frame is reallocated. That is why "blocks in use" and "entries still available for hits" are two different numbers, and why `free_blocks` is called with a request's blocks reversed — tail blocks are the least reusable prefix, so they sit nearest the front of the victim order.

Four analogy breaks are marked. The one that bites a systems reader hardest is granularity: there is no page fault, so when `allocate_slots` returns `None` the scheduler does not service a miss — it preempts an entire sequence, frees every block it holds, and resets `num_computed_tokens` to zero. The unit of eviction is the request. The fourth break is what makes that tolerable: KV is recomputable from the token ids, so a bad eviction costs one prefill and never costs correctness. No storage tier you have run in production is allowed to make that trade.

**What this diagram omits.** The diagram draws ONE KV cache group. Laguna's 12 full_attention + 36 sliding_attention layers actually run as separate groups with different frame geometry over the same pool; that is stated in the REQS subgraph title but not drawn, because drawing it doubles the node count for a second-order point. Frame numbers (12, 19, 31, 44, 88), the two-block shared prefix, and ref_cnt values are illustrative, not from a trace. Everything about the MECHANISM is from source; the specific integers are not. 192.0 KiB/token is the exact worst case if every layer retained every token. Real residency is far lower because 36 of 48 Laguna layers are windowed at 512 tokens (ASSUMPTIONS: kv-per-token-laguna). The diagram gives the ceiling, not the expected footprint. The >=62 GiB fast tier and ~200 GB/s figures are SINGLE-RUN, one run per arm — an anecdote by this lab's own >=3-seed standard. 62 GiB is also a floor (the sweep hit the >=32 GiB single-tensor fault before it found a bandwidth knee), not a measured edge. The >=32 GiB single-tensor hang is real and measured, but its mechanism is an untested [A] (suspected 32-bit overflow in the copy path). The diagram's advice to chunk the pool follows from the symptom, not from a diagnosis. Omitted from the vLLM machinery: hybrid kernel-vs-manager block sizes (worker/block_table.py:64-77), the watermark and reserved_blocks headroom in allocate_slots, fine-grained sub-block hash probing (single_type_kv_cache_manager.py:716-736), the null block, KV connector / external-cache paths, and copy-on-write partial-hit bookkeeping. Each is real; none is needed to understand the block table. The Triton pointer is one of several attention backends. Other backends consume the same block_table_tensor through CommonAttentionMetadata (`memory/vllm/vllm/v1/attention/backend.py:437`) but walk it differently; the diagram shows the unified-Triton path as representative. Line numbers are pinned to the revision recorded in PROVENANCE.md. Re-fetching upstream will move them; scripts/generate_code_map.py is what detects that, and this diagram will need the same treatment since its labels carry pointers.

*Source: [`docs/diagrams/paged-attention-block-table.mmd`](diagrams/paged-attention-block-table.mmd) — 27 grounding references.*

---

## The attribution harness

```mermaid
flowchart TB
    subgraph pre["Precondition: Hardware Validation Gate, still open"]
        det["Determinism proven<br/>one seed, bit-identical logits"]
        fp32["fp32 reference run<br/>bf16 unproven on gfx1151, and a numerics<br/>wobble is indistinguishable from a policy effect"]
        buf["Every buffer under 32 GiB<br/>a 32 GiB tensor hangs silently at 0% CPU"]
    end

    probe["PROBE<br/>one prompt, one fixed seed<br/>the answer is a deterministic function of one injected span,<br/>so the correct attribution is known by construction"]

    subgraph harness["Mnemosyne oracle-diff harness: the lab's deliverable"]
        oracle["ORACLE run<br/>full KV cache, nothing withheld<br/>192.0 KiB/token, so 24.0 GiB at 128k context<br/>against a measured 62 GiB fast tier at about 200 GB/s"]
        treated["TREATED run<br/>same prompt, same seed, some entries withheld"]
        ledger["DROP LEDGER<br/>layer, head and position of every withheld entry"]
        kl["Per-token KL<br/>oracle against treated"]
        loc["LOCALISER<br/>attribute the KL spike at token t to entries withheld from t<br/>attributable only on the 12 full-attention layers: on the 36<br/>sliding layers, a token past the 512 window was unreadable anyway"]
    end

    subgraph nulldist["Seed-to-seed null"]
        nullruns["Same config, same prompt, two different seeds<br/>no policy, no fault"]
        band["Null KL band<br/>the noise floor a spike must clear to count"]
    end

    subgraph calib["PHASE 1: fault injection certifies the EVAL, not a policy"]
        f_absent["needle absent<br/>expect: score falls to chance"]
        f_kvdrop["the needle's own KV entries dropped<br/>expect: large drop, localised to them"]
        f_uniform["uniform eviction of p% of entries<br/>expect: monotone degradation slope"]
        f_rope["cache re-packed so RoPE phase no longer matches position<br/>expect: large drop"]
        f_headmask["retrieval heads masked<br/>expect: retrieval tasks drop, others intact"]
        f_shuffle["haystack sentence order shuffled<br/>expect: NO drop"]
        injector["Fault injector<br/>a KNOWN fault stands in for the policy"]
        check{"All six responses<br/>match prediction?"}
    end

    certified["Instrument certified<br/>it demonstrably detects a memory regression"]
    retire["Eval retired<br/>it was measuring something other than memory"]

    subgraph phase2["PHASE 2: unreachable until Phase 1 passes"]
        policy["Candidate policy P at Mnemosyne's three plug points<br/>write-time admission, deferred eviction, read-time selection"]
        verdict["Report divergence AND downstream task accuracy,<br/>and their correlation, as a first-class result"]
    end

    risk["RISKIEST ASSUMPTION: that divergence localises at all.<br/>It may smear evenly across every dropped entry, or flip one<br/>critical token at negligible mean KL. If so the localiser<br/>returns nothing and this whole plan changes."]

    pre -->|"nothing below is admissible until these hold"| probe
    probe --> oracle
    probe --> treated
    probe --> nullruns
    nullruns --> band
    treated --> ledger
    oracle --> kl
    treated --> kl
    kl --> loc
    ledger -->|"the only admissible causes"| loc
    band -->|"significance threshold"| loc

    f_absent --> injector
    f_kvdrop --> injector
    f_uniform --> injector
    f_rope --> injector
    f_headmask --> injector
    f_shuffle --> injector
    injector -->|"occupies the TREATED slot"| treated

    loc -->|"calibration mode"| check
    check -->|"yes, all six moved as predicted"| certified
    check -->|"no, the eval sat still"| retire
    retire -.->|"redesign the probe, re-inject"| probe
    certified -->|"unlocks"| policy
    policy -->|"now occupies the TREATED slot"| treated
    loc -->|"certification mode"| verdict
    loc -.-> risk

    classDef assumption fill:#fff4d6,stroke:#b8860b,stroke-width:3px;
    classDef pass fill:#e3f5e3,stroke:#2e7d32,stroke-width:2px;
    classDef fail fill:#fbe3e3,stroke:#c62828,stroke-width:2px;
    classDef measured fill:#e8f0fb,stroke:#1a4f8a,stroke-width:2px;
    class risk assumption;
    class certified pass;
    class retire fail;
    class oracle,band measured;
```

Read this as a differential test rig, not a training pipeline. One probe prompt and one fixed seed drive two forward passes that differ in exactly one thing: what the cache was allowed to keep. The oracle run keeps everything — affordable here and almost nowhere else, because 192.0 KiB/token of KV at the reference geometry is 24.0 GiB at 128k context against a measured 62 GiB fast tier. The treated run withholds entries and logs precisely which ones. Per-token KL between the two output distributions is the signal, the drop ledger is the suspect list, and the localiser tries to join them — with the seed-to-seed null band deciding what counts as a spike at all.

The non-obvious part is the left-hand loop, and it is why this is a system rather than a script. Phase 1 never judges a policy. It substitutes six *known* faults for the policy, each with a response predicted in advance, and asks whether the eval moves. An eval that survives all six without moving is measuring something other than memory and gets retired. This is ordinary alerting discipline: you do not trust a pager you have never seen fire. Only a certified instrument is admitted to Phase 2.

Two places the systems analogy breaks. First, there is no request ID. Causality runs through a continuous attention distribution rather than a call graph, so attribution is a statistical join against a noise floor, not a trace lookup — which is exactly the marked assumption: divergence may smear evenly across every dropped entry. Second, 36 of the 48 reference layers window at 512 tokens, where an out-of-window entry is architecturally unreadable and dropping it is lossless by construction. Flat KL on those layers is not evidence the policy was harmless; it is evidence you dropped something the model could never have read.

**What this diagram omits.** Nothing in this diagram has been run. The Hardware Validation Gate is open, `bf16-numerics-unproven` is untested, and determinism has not been demonstrated on this stack — the entire left column is a precondition, not a completed step. The KV geometry in the ORACLE box (192.0 KiB/token, 24.0 GiB at 128k, 12 full / 36 sliding layers, 512 window) is the Laguna S 2.1 *reference* model's, read from its config. The lab's own ablations run at 20M–300M params, where the absolute numbers differ; the ratio argument is what transfers. The ≥62 GiB fast tier at ~200 GB/s is a single run per arm — an anecdote by the house standard (`gpu-fast-tier-size`). Its upper edge is unmeasured; the sweep stopped at the 32 GiB single-tensor fault, not at a bandwidth knee. The diagram draws one probe and one pair of runs. The experimental standard is ≥3 seeds with confidence intervals; a single oracle/treated pair is an anecdote and the null band needs many pairs to exist at all. The six faults are checked here against the localiser's output for compactness. The §6.2 protocol also checks the eval's own score (chance level, monotone slope, targeted vs global drop); the diagram collapses those two checks into one `check` node. Phase 1 is shown as a clean pass/fail. In practice a partial pass — say, five of six faults detected — is the likely and more awkward outcome, and the diagram says nothing about what happens then. Omitted: the cost of the instrument itself. Reading device tensors to log per-token KL stalls the GPU pipeline (CODE_MAP, OLMo-core trainer.py:1037/:1394), so attribution telemetry is not free the way production logging is. Omitted: the internals and ordering of Mnemosyne's three plug points, and the policy-decomposition step (observation window / scoring function / budget allocation / sink pinning) that follows a verdict. The governing ADR (`attribution-instrument-over-eviction-policy`) is `Proposed`, not Accepted — it awaits founder review alongside `research/synthesis.md`. This diagram documents a proposed design.

*Source: [`docs/diagrams/attribution-oracle-diff.mmd`](diagrams/attribution-oracle-diff.mmd) — 18 grounding references.*

---
