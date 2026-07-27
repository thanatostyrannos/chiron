---
title: Running Laguna XS 2.1 locally â€” the download you have not paid for, and what the config already tells you
version: 1.0.0
date: 2026-07-26
track: F â€” Inference
prereqs: attention-variants-and-kv-cost, kv-cache-mechanics, hybrid-attention-and-ratios, moe-and-routing, tokenization
recommended: paged-attention-and-prefix-reuse, measuring-memory
difficulty: medium â€” the arithmetic is exact and closes to the byte, but the honesty discipline is the hard part: most of this module is about knowing which numbers you have and which you only think you have
time: 2â€“3 h reading; 1.5â€“2 h for the three exercises, all of which run **today, without weights**, on CPU or GPU
---

# Running Laguna XS 2.1 locally

**A notation warning, first, because it changes how you read every number below.**
In this curriculum `[M]` means *measured*. In this module every `[M]` is a value **read out
of a pinned artifact** â€” `config.json`, `tokenizer.json`, `model.safetensors.index.json`,
`chat_template.jinja`, or source at a revision recorded in `research/reference/PROVENANCE.md`
â€” which is deterministic and reproducible by anyone with the same revision. That is the same
convention `ASSUMPTIONS.md` uses for its `reference-model`, `laguna-heads-uniform` and
`kv-per-token-laguna` rows.

**There is not one runtime measurement of my own in this module.** The weights are not
downloaded. I did not run the exercises. Every timing, throughput and memory-at-runtime
figure below is *derived arithmetic over someone else's `[M]`*, and is tagged `[A]` with the
inputs named. Two earlier modules in this curriculum were burned by tagging a
non-reproducing observation `[M]`; this module's answer to that is to make the boundary
loud rather than to hedge each sentence.

**Difficulty and time, honestly.** The math here is multiplication. What makes the module
worth two evenings is that it is the first one where the artifact under study is *the actual
reference model* rather than a shape derived from it, and the artifact disagrees with its own
documentation in three separate places. Budget 2â€“3 hours for sections 2â€“5 with the config
open in another window, and 90 minutes for the exercises. All three exercises run with what
is already installed (`torch`, `numpy`, `jinja2`) plus the Python standard library. None of
them needs the weights.

**Prerequisites, and what this module refuses to re-teach.** You need
`attention-variants-and-kv-cost.md` for `2Â·LÂ·n_kvÂ·d_hÂ·b`, `kv-cache-mechanics.md` for the
three budgets (residency / read traffic / maintenance traffic) and for why residency is a
sum over layer types rather than a product with `L`, `hybrid-attention-and-ratios.md` for
why a 3:1 SWA:global schedule is the interesting object, `moe-and-routing.md` for sigmoid
routing and the correction bias, and `tokenization.md` for byte-level BPE. None of that is
restated. This module is the operational one: *which file do you fetch, what does it cost,
what can you check before it arrives, and what do you look at the moment it lands.*

---

## 1. What this module settles

**One:** the entire memory budget for running Laguna XS 2.1 is computable from three files
you already have on disk â€” the parameter count derived from `config.json` alone comes to
**33,442,617,088 parameters, which is `total_size / 2` in the safetensors index to the
byte** `[M]`, and the KV growth slope comes to **40,960 bytes per token, not the 163,840 the
top-level config implies** `[M]`. **Two:** the bf16 checkpoint is **62.29 GiB**, against a
fast tier measured good to **â‰¥62 GiB** `[M]` (`ASSUMPTIONS.md â†’ gpu-fast-tier-size`) â€” so on
this machine the full-precision artifact consumes the entire characterised fast tier and
leaves nothing for KV or activations, which makes the choice of *quantised artifact* a
research-design decision rather than a convenience. **Three:** two shipped implementations
of this same model disagree about what `gating=True` means, in a way that silently changes a
projection's width by 128Ã— and that the shipped `config.json` happens to dodge â€” which is
the module's standing lesson: **the config is the contract, the code is the behaviour, and
on this model they are not the same document.**

---

## 2. Theory in plain language

### 2.1 "Running a model locally" is four different things

A model repository is not a program. It is a **spec plus a blob**, and four different
runtimes will read the same spec and produce four different memory systems from it. You
already know this shape: an OCI image is a manifest plus layers, and `docker run`,
`containerd`, `crun` and Kata will all honour the manifest while giving you wildly different
isolation, page-cache behaviour and startup cost. Same here.

| Artifact | What it is | Present on disk now? |
|---|---|---|
| `config.json` | the architecture spec â€” 229 lines that fully determine parameter count, KV geometry, and the per-layer attention schedule | **yes** |
| `tokenizer.json` + `tokenizer_config.json` | the textâ†”id contract, including the chat scaffolding tokens | **yes** |
| `chat_template.jinja` | the prompt-construction program, in Jinja | **yes** |
| `modeling_laguna.py` + `configuration_laguna.py` | the reference forward pass, shipped as remote code | **yes** |
| `model-*-of-00014.safetensors` | 62.29 GiB of numbers | **no â€” LFS pointers only** |
| `generation_config.json` | default sampling + a speculative-decoding config | **yes** |

Everything except the last row's blob is here. That is not a degraded situation; it is
most of the information. The blob determines *how well* the model works. The spec determines
*what it costs to run*, and cost is what this lab studies.

> **Systems bridge.** Treat the repo as an image manifest with the layers not yet pulled.
> You can compute the pull size, the on-disk footprint, the layer count, and the
> dependency graph from the manifest alone, and you routinely do that before deciding
> whether to pull.
>
> **Where it breaks, and it breaks hard.** A container manifest is *authoritative* about its
> layers: the digests are content-addressed, so the manifest cannot lie about what you will
> get. `config.json` is not content-addressed against the code that reads it. It is a bag of
> keyword arguments handed to a Python constructor, and the constructor supplies defaults
> for anything absent. **A key you omit is not an error; it is a different model.** Â§2.5
> gives three live instances in this one repository. There is no equivalent failure in an
> OCI registry, and it is the single most important adjustment to make.

### 2.2 The download is a real cost line and it has not been paid

`[M]` `model.safetensors.index.json:3` gives `total_size: 66885234176` bytes.

```
66,885,234,176 B  =  62.2917 GiB  =  66.885 GB
14 shards, each ~4.768 GiB   (model-00001-of-00014.safetensors:3 â†’ size 5120041576)
```

`[A]` Transfer time is `66.885 GB Ã— 8 / link_Gbit_per_s` seconds: **~9 minutes at 1 Gbit/s,
~18 minutes at 500 Mbit/s, ~89 minutes at 100 Mbit/s.** Arithmetic, not a measurement â€” your
link is the variable.

**Disk is the trap, not bandwidth.** Two mechanisms duplicate the payload:

- A plain `git clone` of a git-lfs repo writes each object *twice*: once into
  `.git/lfs/objects/`, once into the working tree. That is **~124.6 GiB peak**. The clone we
  already have avoided this because the fetch script set `GIT_LFS_SKIP_SMUDGE=1` â€” which is
  exactly why the `*.safetensors` files on disk are three-line pointer stubs
  (`model-00001-of-00014.safetensors:1-3`).
- `hf download` without `--local-dir` writes into the hub cache and then materialises the
  working copy. On Linux that is a symlink and costs nothing; on Windows, symlink creation
  requires Developer Mode or elevation, and the fallback is a **copy**.

`[A]` High confidence, cheapest check is `Get-PSDrive C` before and after a one-shard test
download: **budget 2Ã— the download size in free space unless you have verified otherwise.**

**The exact commands.** `ENVIRONMENT.md` currently records `huggingface-cli` as the one
failing preflight check, so step zero is real:

```powershell
. .\scripts\activate-lab.ps1
pip install "huggingface_hub[cli]"

# Newer huggingface_hub exposes this as `hf`; older versions as `huggingface-cli`.
# Both accept the same arguments here.
hf download poolside/Laguna-XS-2.1 --local-dir C:\models\Laguna-XS-2.1
```

Three constraints on that path, none of them cosmetic:

1. **Not inside the repo.** Hard rule 5: no weights in git, nothing over 20 MB. The clone
   at `research/reference/models/laguna-xs/` is gitignored except `*.md`, but putting 62 GiB
   under a repo root you routinely `git add -A` in is a mistake waiting for a distracted
   evening.
2. **No spaces in the path.** `ENVIRONMENT.md` checks this explicitly and the lab venv lives
   at `C:\venvs\lab` for the same reason â€” spaces are known-broken on this pip/ROCm stack.
   `C:\models\...` matches the established convention.
3. **Credentials are not yours to supply** (CLAUDE.md). The config/tokenizer clone succeeded
   with no auth `[M]` (`PROVENANCE.md`, revision `205dc65dd4bd`), but the model card carries
   an `extra_gated_description` in its frontmatter (`README.md:4-6`), so the LFS objects may
   sit behind an accepted-license click. **If `hf download` asks for a login, stop and ask
   the owner to run `hf auth login`.** Do not work around it.

**The smaller artifacts, and why you probably want one of them.** `[M]` The model card lists
FP8, NVFP4 and INT4 variants and an official GGUF repo (`README.md:35`, `README.md:276`).
`[A]` The Q4_K_M GGUF is roughly **17.5 GiB** â€” derived, not measured: 33.44e9 parameters at
the 4.5 bits/weight that `block_q4_K` actually costs (`[M]` 144 bytes per 256 weights, read
from `ggml-common.h:327` in `research/notes/inference-and-quantization.md`). The true file
is larger, because Q4_K_M keeps selected tensors at 6 bits and does not quantise embeddings
or the output projection. Â§3.5 shows why that 17.5-vs-62.29 difference is the whole decision.

### 2.3 What you can settle before the blob arrives

Four things, and they are the four that matter for a memory lab:

1. **The exact parameter count and the exact active-parameter count** â€” Â§3.2, Â§3.3, and it
   closes to the byte against the index.
2. **The exact KV geometry and growth slope** â€” Â§3.4. This is the number that sizes every
   long-context experiment.
3. **The prompt-construction contract and its token-level consequences** â€” Â§3.6 and
   Exercise B. Prefix-cache behaviour is decided here, before any weight is loaded.
4. **Whether the model will fit, and in which artifact** â€” Â§3.5, cross-referenced to
   `ASSUMPTIONS.md â†’ gpu-fast-tier-size`.

### 2.4 The reference model is a two-tier cache that cannot promote

You know the shape from `hybrid-attention-and-ratios.md`; here it is instantiated. `[M]`
`config.json:59-99` gives 40 entries in `layer_types`: `full_attention` at indices
0, 4, 8, 12, 16, 20, 24, 28, 32, 36 â€” **10 global layers** â€” and `sliding_attention`
everywhere else â€” **30 windowed layers**, `sliding_window: 512` (`config.json:41`). A strict
GSSS period-4 pattern, the advertised 3:1 ratio, read from the artifact rather than quoted
from the card.

`[M]` And the layers are structurally different, not the same block with a different mask:
`config.json:187-228` gives `num_attention_heads_per_layer` = **48 query heads on the ten
global layers, 64 on the thirty windowed ones**, while `num_key_value_heads` is a scalar 8
with no per-layer override. So the GQA group size `G = n_q/n_kv` is **6 on global layers and
8 on windowed ones**, and the top-level `num_attention_heads: 48` (`config.json:14`) is wrong
for 30 of 40 layers.

> This is the XS-scale version of `ASSUMPTIONS.md â†’ laguna-heads-uniform`, which recorded
> 48/72 for Laguna-S. **XS is 48/64.** Note what does *not* change: `n_kv` is uniform at 8 on
> both models, so the KV product is unaffected and `kv-per-token-laguna`'s reasoning carries
> over unchanged. What changes is the arithmetic intensity, which is per-layer-type on both.

> **Systems bridge.** llama.cpp allocates this as two literal cache objects â€” a full-size one
> for the global layers and one sized `n_swa + n_ubatch` for the windowed ones
> (`llama-kv-cache-iswa.cpp:73`). That is capacity planning across two storage classes and
> you have done it a hundred times.
>
> **Where it breaks â€” three ways, and the third is the one people miss.** (a) There is no
> promotion, demotion or miss path: a layer is bound to a tier forever by its index, decided
> once at load. (b) "Eviction" from the small tier is not a policy with a hit-rate cost,
> because out-of-window tokens are architecturally unreadable â€” discarding them is *lossless*
> rather than a gamble. (c) `[M]` The two tiers are not numerically interchangeable:
> `config.json:42-57` gives the global layers YaRN RoPE at `Î¸=500000` over half the head dims
> (`partial_rotary_factor: 0.5`, `factor: 32.0`) and the windowed layers **plain RoPE at
> `Î¸=10000` over all 128 dims**. You cannot "just widen the windows" to test long context â€”
> the windowed layers were never trained with positional encoding that reaches past 512.

### 2.5 Two implementations of one model, and three places they disagree

Both are on disk. `research/reference/models/laguna-xs/modeling_laguna.py` is the remote code
shipped *with the checkpoint*; `research/reference/architecture/transformers/src/transformers/models/laguna/modeling_laguna.py`
is the upstream-merged version. Same model. Read them side by side and you find:

**(a) `gating=True` means opposite things.** `[M]` Upstream's config docstring
(`architecture/transformers/src/transformers/models/laguna/configuration_laguna.py:33`) says
"``True`` or ``"per-head"`` applies one gate per head", and the code implements it
(`.../modeling_laguna.py:370`: `self.gate_per_head = config.gating is True or config.gating == "per-head"`).
The shipped remote code's docstring (`models/laguna-xs/configuration_laguna.py:37-42`) says
"When ``True`` or ``"per-element"`` a g_proj linear layer with output size
``num_attention_heads * head_dim`` is added", and the code implements *that*
(`models/laguna-xs/modeling_laguna.py:394`: `self.gate_per_head = gating == "per-head"`).
Both default `gating: bool | str = True`.

So a config that says `gating=True` builds `g_proj` with output width **64** under upstream
and **8192** under the shipped code, on a 64-head layer. A 128Ã— difference in that
projection's parameter count, from a default. Laguna-XS 2.1 dodges it entirely because
`config.json:40` says `"gating": "per-head"` â€” but that is luck, and Proteus will write its
own configs.

**(b) A missing key silently disables a feature.** `[M]`
`models/laguna-xs/configuration_laguna.py:189-193` derives `swa_rope_parameters` from
`rope_parameters["sliding_attention"]` and says why in a comment: *"config.json stores SWA
rope nested in rope_parameters['sliding_attention'] and carries no top-level
swa_rope_parameters. Derive it here, else the sliding-window layers silently reuse the
full-attention rope."* One `if` between correct behaviour and thirty layers running the wrong
positional encoding, with no error.

**(c) The layer schedule is a list in one implementation and a modulo in the other.** `[M]`
HuggingFace reads `config.layer_types[layer_idx]` (`models/laguna-xs/modeling_laguna.py:375`).
llama.cpp calls `set_swa_pattern(4, dense_first=true)` (`llama-cpp-laguna/src/models/laguna.cpp:41`)
and *recomputes* the schedule as `il % 4 == 0`. For the shipped config these agree exactly.
For any non-periodic schedule â€” the first thing a hybrid-ratio ablation will produce â€” they
do not, and llama.cpp will not tell you.

> **Systems bridge, and it is the right one.** This is a striping function versus a
> placement map. `il % 4` is `hash(block) % ndisks`; `layer_types[i]` is a lookup table.
> You have made this exact trade in a storage layer and you know the failure mode: the
> striping function is cheap and self-describing right up until someone needs a placement
> that the function cannot express, and then it is a silent data-placement bug rather than
> an error.
>
> **Where it breaks.** In a storage layer you would detect the divergence on the first read,
> because the data would be wrong. Here the model still produces fluent text. A wrong layer
> schedule degrades long-range recall by a few points on a benchmark you were not running.
> **There is no read-verify path.** That is the standing reason this lab budgets for
> attribution instrumentation rather than outcome metrics.

---

## 3. The math that actually matters

### 3.1 Symbols

| Symbol | Reads as | Source in `config.json` | Value for Laguna-XS 2.1 |
|---|---|---|---|
| `L` | total decoder layers | `num_hidden_layers` | 40 |
| `L_g`, `L_w` | global / windowed attention layers | counted from `layer_types` | 10 / 30 |
| `w` | sliding window, in tokens | `sliding_window` | 512 |
| `n_kv` | key-value heads (uniform) | `num_key_value_heads` | 8 |
| `n_q,l` | query heads in layer `l` | `num_attention_heads_per_layer` | 48 global / 64 windowed |
| `d_h` | head dimension | `head_dim` | 128 |
| `d` | model width | `hidden_size` | 2048 |
| `b` | bytes per stored KV element | from `torch_dtype` | 2 (bf16) |
| `V` | vocabulary size | `vocab_size` | 100,352 |
| `E` | routed experts per MoE layer | `num_experts` | 256 |
| `k` | experts selected per token | `num_experts_per_tok` | 8 |
| `d_e` | expert intermediate width | `moe_intermediate_size` | 512 |
| `d_s` | shared-expert intermediate width | `shared_expert_intermediate_size` | 512 |
| `d_ff` | dense-MLP intermediate width | `intermediate_size` | 8192 |
| `c` | **bytes per token per layer** = `2Â·n_kvÂ·d_hÂ·b` | derived | **4096 B = 4 KiB** |
| `T` | tokens currently in context | runtime | â‰¤ 262,144 |

`c` is identical to Laguna-S's, because `n_kv` and `d_h` are identical. Everything in Â§3.4 is
that one number times a count.

### 3.2 Parameter count from the config alone â€” and it closes to the byte

This is worth doing by hand once, because the check at the end is unusually strong.

**Attention, per layer.** With `n_q` query heads:

```
q_proj : d Ã— (n_qÂ·d_h)          k_proj : d Ã— (n_kvÂ·d_h)      v_proj : d Ã— (n_kvÂ·d_h)
o_proj : (n_qÂ·d_h) Ã— d          g_proj : d Ã— n_q             (per-head gating)
q_norm : d_h                    k_norm : d_h                 (QK-norm, head_dim-sized)
```

In words: the query projection maps the 2048-wide residual stream to `n_q` heads of 128
dims; the key and value projections map it to only 8 heads each â€” that asymmetry *is* GQA;
the output projection maps back; `g_proj` emits one scalar per head because gating is
per-head; and the two norms are 128-element vectors applied inside each head before RoPE.

```
global  (n_q=48):  12,582,912 + 2,097,152 + 2,097,152 + 12,582,912 + 98,304 + 256 = 29,458,688
windowed(n_q=64):  16,777,216 + 2,097,152 + 2,097,152 + 16,777,216 + 131,072 + 256 = 37,880,064

attention total  =  10 Ã— 29,458,688  +  30 Ã— 37,880,064  =  1,430,988,800
```

**MoE, per sparse layer.** `[M]` `mlp_only_layers: [0]` and `decoder_sparse_step: 1`
(`config.json:27-30`) put a dense MLP on layer 0 and MoE on layers 1â€“39, which
`mlp_layer_types` (`config.json:102-143`) states redundantly and identically. Each MoE layer:

```
router gate      : d Ã— E                    = 2048 Ã— 256          =       524,288
correction bias  : E                                              =           256
shared expert    : 3 Ã— d Ã— d_s              = 3 Ã— 2048 Ã— 512      =     3,145,728
routed experts   : E Ã— 3 Ã— d Ã— d_e          = 256 Ã— 3,145,728     =   805,306,368
                                                        per layer =   808,976,640
                                                             Ã— 39 = 31,550,088,960
```

The `3 Ã—` is gate/up/down â€” SwiGLU has three matrices, not two.

**Everything else.**

```
dense MLP, layer 0 :  3 Ã— d Ã— d_ff  = 3 Ã— 2048 Ã— 8192  =      50,331,648
embed_tokens       :  V Ã— d         = 100,352 Ã— 2048   =     205,520,896
lm_head            :  V Ã— d         (tie_word_embeddings: false)  = 205,520,896
layernorms         :  40 Ã— 2 Ã— d  +  d                 =         165,888
```

**Sum:**

```
  1,430,988,800   attention
+31,550,088,960   MoE
+    50,331,648   dense MLP
+   205,520,896   embeddings
+   205,520,896   lm_head
+       165,888   norms
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
 33,442,617,088   parameters
```

`[M]` And `model.safetensors.index.json:3` says `total_size: 66,885,234,176` bytes.
`66,885,234,176 / 2 = 33,442,617,088`. **Exact, to the parameter.**

Two things fall out of an exact match that a 0.1% match would not give you. First, **every
tensor in this checkpoint is bf16** â€” including the 256-element router correction bias and
the 2048-element layernorms. Nothing is kept in fp32. Second, the decomposition above is
*correct*, not merely plausible: had `tie_word_embeddings` been true, or had the shared
expert been counted wrong, or had `e_score_correction_bias` been per-layer-per-head instead
of per-expert, the sum would miss. This is the strongest form of config verification
available without weights, it takes twenty lines of Python, and it is Exercise A.

### 3.3 Active parameters, and the decode floor

MoE's whole proposition is that you *hold* 33B and *read* far less per token. Which experts
are active changes every token, so residency is all 33B; traffic is not.

```
attention                          1,430,988,800     (all of it, every token)
per MoE layer: 524,288 + 256 + 3,145,728 + 8Ã—3,145,728  =  28,836,096
      Ã— 39                         1,124,607,744
dense MLP layer 0                     50,331,648
lm_head                              205,520,896     (you need every logit)
norms                                    165,888
embed_tokens                             (a 4 KiB row gather â€” negligible)
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
active per token                   2,811,614,976     â‰ˆ 2.81 B
```

`[M]` The model card says "3B activated per token" (`README.md:43`). Our decomposition says
2.81B. The gap is a convention question the card does not state â€” exclude `lm_head` and you
get 2.61B; count `embed_tokens` in full and you get 3.02B. **Do not carry the card's round
number into a cost model; carry the decomposition.**

**The ratio that defines this model on this machine:**

```
resident bytes / bytes-read-per-token  =  33,442,617,088 / 2,811,614,976  =  11.89Ã—
```

You must hold **62.29 GiB** to move **5.24 GiB** per decode step. That is a *capacity*
problem wearing a bandwidth problem's clothes, and it is precisely why this lab bought
128 GB of unified memory instead of a 20 GB discrete card.

`[A]` **The decode floor.** 2,811,614,976 params Ã— 2 B = 5,623,229,952 B = 5.623 GB. At the
`[M]` 199.9 GB/s device-to-device figure (`ASSUMPTIONS.md â†’ gpu-fast-tier-size`):

```
weights-only step  =  5.623 GB / 199.9 GB/s  =  28.13 ms  â†’   35.6 tok/s
+ KV at T = 32,768 (Â§3.4: 1.405 GB)          =   7.03 ms
                                       total â‰ˆ 35.16 ms   â†’   28.4 tok/s
```

This is an **upper bound and a soft one**. Three reasons it will not be met, in descending
order of size:

1. **The expert read is a gather, not a stream.** `[M]` The safetensors index stores experts
   *unfused*: `model.layers.<i>.mlp.experts.<e>.{gate,up,down}_proj.weight`, one tensor per
   expert per projection. That is `39 Ã— 256 Ã— 3 = 29,952` separate 2 MiB tensors, and a
   decode step touches `39 Ã— 8 Ã— 3 = 936` of them chosen by the router. The `[M]` 199.9 GB/s
   was measured on a contiguous device-to-device copy; nothing in `ASSUMPTIONS.md`
   characterises a 936-way scatter-gather of 2 MiB blocks, and
   `kv-cache-mechanics.md` Exercise B already found that a *decode-shaped* attention read
   tops out around **150 GB/s**, 75% of the copy figure.
2. **The KV read is not the copy benchmark either** â€” same finding, same module.
3. `[M]` GEMM on this machine reaches 20.9 TFLOPS bf16, 63% of the figure cited for the
   silicon, unexplained (`ASSUMPTIONS.md â†’ gemm-throughput-below-reference`). At batch 1 that
   barely matters â€” the ridge point is ~105 FLOP/byte and decode sits at single digits â€” but
   it matters for prefill.

> **Systems bridge, and this one is nearly exact.** A sparse MoE at batch 1 is a
> **cache-hostile hash join**: you hold the whole build side resident (33B parameters) and
> probe 936 randomly-selected 2 MiB partitions per output row. You already know that the
> throughput of such a thing is governed by the *number and locality of the probes*, not by
> the size of the build side.
>
> **Where it breaks.** In a hash join you would fix this by partitioning to fit cache, or by
> batching probes so each partition is touched once. Batching does work here â€” it is exactly
> what raises the weight-read arithmetic intensity from `2B/b_w` â€” but it works *only on the
> weight term*, and it makes the expert selection **less** local, not more, because a batch
> of 64 tokens will collectively activate a large fraction of all 256 experts per layer. The
> partition-locality move you would reach for first makes the problem worse. That inversion
> is the interesting part.

### 3.4 KV residency: the slope, the intercept, and the parity point

From `kv-cache-mechanics.md` Â§3.2, residency is a sum over layer types:

```
R(T)  =  c Â· L_g Â· T   +   c Â· L_w Â· min(T, w)
```

In words: every global layer holds a 4 KiB entry for every token ever seen; every windowed
layer holds at most `w` of them and stops growing the moment `T` passes 512.

`[M]` Substituting Laguna-XS 2.1:

```
growing slope  =  c Â· L_g       =  4096 Ã— 10  =    40,960 B/token  =  40 KiB/token
fixed term     =  c Â· L_w Â· w   =  4096 Ã— 30 Ã— 512  =  62,914,560 B  =  60 MiB
nominal (if every layer were global) = c Â· L = 4096 Ã— 40 = 163,840 B/token = 160 KiB/token
```

| Context `T` | growing | fixed | `R(T)` | all-global | saving |
|---|---|---|---|---|---|
| 512 | 20.0 MiB | 60 MiB | 80.0 MiB | 80.0 MiB | 1.00Ã— |
| 1,536 | 60.0 MiB | 60 MiB | 120.0 MiB | 240.0 MiB | 2.00Ã— |
| 32,768 | 1.250 GiB | 60 MiB | **1.309 GiB** | 5.000 GiB | 3.82Ã— |
| 131,072 | 5.000 GiB | 60 MiB | **5.059 GiB** | 20.00 GiB | 3.95Ã— |
| 262,144 (max) | 10.00 GiB | 60 MiB | **10.059 GiB** | 40.00 GiB | 3.98Ã— |

**Byte parity** â€” where the windowed layers and the global layers hold equal bytes â€” is at

```
T*  =  (L_w / L_g) Â· w  =  3 Ã— 512  =  1,536 tokens
```

and note that it depends **only on the ratio and the window**, not on `L`, `n_kv`, `d_h` or
`b`. Laguna-S's parity point is also 1,536, for the same reason. Below 1,536 tokens the
hybrid saves you nothing and its constant term dominates â€” which is exactly the regime most
small-scale experimentation lives in, so keep it in mind before concluding that a windowing
ablation "did nothing."

**The slope is the measurable signature and Exercise C measures it.** If you instrument a
real run and the residency slope comes out at 163,840 B/token instead of 40,960, the
`layer_types` list did not survive whatever path you took to get there. `[M]`
llama.cpp makes that failure explicit: `laguna.cpp:41` skips the entire hybrid path and
builds an all-full-attention model if the `sliding_window` GGUF key is absent â€” no error, a
4Ã— residency increase, and a model that still generates fluent text.

`[M]` The model card advertises an **FP8 KV cache** as a headline feature (`README.md:33`).
That halves `c` to 2048 B, taking the slope to 20 KiB/token and the fixed term to 30 MiB.
`[M]` Two caveats from this lab's own record before you build a budget on it: the KV dtype is
a property of the *inference path*, not of the model â€” the Laguna llama.cpp branch has no FP8
at all, and its quantised-KV story is `q8_0`-style block quantisation at 8.5 bits/element
plus an automatic Hadamard rotation (`llama-kv-cache.cpp:319`) â€” and `torch._scaled_mm` is
unsupported on gfx1151 `[M]`, so an FP8 cache under PyTorch here must dequantise to bf16
before every attention call, which `kv-cache-mechanics.md` Exercise C measured at **2.9â€“3.1Ã—
slower** than plain bf16. FP8 KV on this machine is a capacity win and a bandwidth loss.

### 3.5 The budget table â€” which artifact you can actually run

`[M]` The binding number is `ASSUMPTIONS.md â†’ gpu-fast-tier-size`: **â‰¥62 GiB at ~200 GB/s**,
with the BIOS UMA carve-out at 96 GB. It is a *floor* â€” the sweep in
`notebook/uma-carveout-controls-fast-tier.md` never found the degradation edge, it hit the
`large-tensor-fault-32gib` fault first â€” so "62 GiB" means "still fast at 62", not "slow at
64". Treat everything above 62 GiB as uncharacterised, not as unavailable.

`[M]` The second binding number is host RAM: with 96 GB carved out, Windows sees **31.6 GB**
(`ENVIRONMENT.md`).

| Artifact | Weights | + KV @ 32k | + KV @ 262k | Fits under the â‰¥62 GiB floor? |
|---|---|---|---|---|
| bf16 safetensors | **62.29 GiB** | 63.60 GiB | 72.35 GiB | **No headroom at all.** Weights alone are 100.5% of the characterised tier. |
| FP8 variant `[A]` â‰ˆ31.15 GiB | 31.15 GiB | 32.46 GiB | 41.21 GiB | Yes, comfortably |
| Q4_K_M GGUF `[A]` â‰ˆ17.5 GiB | 17.5 GiB | 18.8 GiB | 27.6 GiB | Yes, with ~34 GiB spare |

`[A]` on the FP8 and Q4_K_M rows: both are `params Ã— bits/8` arithmetic, and both understate
the real file because neither format quantises everything. Cheapest test is `hf download`
of the index file alone for each variant and reading `total_size`, which costs a few
kilobytes.

**Three hazards this table does not show, all of them `[M]`-backed:**

1. **The prefill score matrix, and it is decisive.** `ASSUMPTIONS.md â†’ sdpa-is-memory-efficient`
   records that on this stack `F.scaled_dot_product_attention` falls back to the math backend
   and **materialises the `BÂ·n_qÂ·TÂ·T` score matrix**, and that
   `torch.backends.cuda.flash_sdp_enabled()` returns `True` anyway. For one 64-head windowed
   layer at batch 1, bf16, the score tensor is `64 Â· TÂ² Â· 2` bytes:

   ```
   T =  4,096  â†’   2.0 GiB      T = 16,384  â†’  32.0 GiB   â† exactly the fault threshold
   T =  8,192  â†’   8.0 GiB      T = 32,768  â†’  128.0 GiB
   ```

   `[M]` `ASSUMPTIONS.md â†’ large-tensor-fault-32gib`: a 32 GiB single tensor **hard-hangs at
   0% CPU** with no error. `[A]` So a bf16 transformers prefill of Laguna-XS at
   **T â‰¥ 16,384 is predicted to hang rather than fail** â€” high confidence in the arithmetic,
   untested because the weights are absent, and the windowed layers hit it *first* despite
   being the cheap ones, because they carry more query heads. Mitigations, in order of
   preference: use llama.cpp (which tiles and never materialises `TÂ²`); chunk the prefill;
   or set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, which `[M]` removes the
   materialisation (147.2 â†’ 6.6 bytes/TÂ² retained) but is marked experimental and is
   therefore a **numerics change** that `scripts/activate-lab.ps1` deliberately leaves off.
2. **Host RAM during load.** Any path that builds the full state dict on CPU before moving
   it needs 62.29 GiB against 31.6 GB available. `device_map="auto"` with
   `low_cpu_mem_usage` streams shard by shard (~4.77 GiB each) and is fine; a naive
   `torch.load` of a merged checkpoint is not. This is a Z13-specific consequence of the
   96 GB carve-out and it is the kind of thing that shows up as an unexplained hang.
3. **No single tensor is dangerous.** Largest is a global layer's K at full context:
   `8 Ã— 262,144 Ã— 128 Ã— 2 = 512 MiB`. The largest weight tensor is bounded by the 4.77 GiB
   shard size. Both far under the 32 GiB fault. The danger is transient activations, not
   parameters.

### 3.6 The prompt is a data structure, and it decides your prefix-cache hit rate

`[M]` `chat_template.jinja:3` emits `ã€ˆ|EOS|ã€‰` as the very first thing in every rendered
prompt. `[M]` `tokenizer_config.json:564` and `:567` both name `ã€ˆ|EOS|ã€‰`: **BOS and EOS are
the same token**, id 2. `[M]` And `tokenizer.json:665-717` installs a `TemplateProcessing`
post-processor that *also* prepends `ã€ˆ|EOS|ã€‰` to every encode.

`[A]` High confidence, and the cheapest test is three lines: **render-then-encode double-prepends
BOS**. `apply_chat_template(..., tokenize=True)` handles it by encoding with
`add_special_tokens=False`; the two-step idiom `text = apply_chat_template(..., tokenize=False)`
followed by `tokenizer(text)` does not. Check `ids[:2] == [2, 2]`.

`[M]` The chat scaffolding is **partly** single tokens. From `tokenizer_config.json:515-562`,
these are added tokens: `<think>`=18, `</think>`=19, `<assistant>`=23, `</assistant>`=24,
`<tool_call>`=25, `</tool_call>`=26. `<system>`, `<user>`, `<tool_response>`, `<arg_key>` and
`<arg_value>` are **not** in the vocabulary at all â€” they are ordinary text that BPE will
split into several tokens each. Two consequences:

- `[M]` All six added tokens carry `"special": false` (e.g. `tokenizer_config.json:521`).
  `[A]` Therefore `decode(..., skip_special_tokens=True)` â€” which is exactly what the model
  card's own example does (`README.md:217`) â€” will **not** strip them, and a `</think>` will
  appear in your output string. Cheapest test: decode a hand-built id list containing 19.
- `[M]` `eos_token_id: [2, 24]` (`config.json:32-35`). Generation stops on `ã€ˆ|EOS|ã€‰` *or*
  `</assistant>`. Two stop conditions, one of which is a structural tag.

`[M]` **The thinking toggle changes exactly one token.** `chat_template.jinja:86-92`: with
`add_generation_prompt=True`, the prompt ends `<assistant>` + `<think>` when thinking is on
and `<assistant>` + `</think>` when it is off. Both are single tokens (18 and 19). So
**toggling thinking does not change the prompt length; it changes the last token id.**

> **Systems bridge â€” prefix caching.** From `paged-attention-and-prefix-reuse.md`: a prefix
> cache keys on a *chain* hash over block-aligned token runs, and the loop breaks at the first
> miss. A change to the last token invalidates only the last partial block. A change in the
> middle invalidates everything downstream.
>
> **Where it breaks for this model specifically.** The card's recommended usage is
> **preserved thinking**: reasoning content from prior assistant turns is re-rendered into
> the history (`README.md:308-310`, `chat_template.jinja:54-58`). That means every turn's
> prompt contains the *middle* of the previous turn's reasoning. Toggling `enable_thinking`
> between turns therefore rewrites the middle of the prefix, not the tail â€” and per
> `chat_template.jinja:55` vs `:57`, turning thinking off replaces `<think>â€¦</think>` with a
> bare `</think>` in **every historical assistant message at once**. The cheap-looking flag
> is a full prefix-cache invalidation for the whole conversation. Nothing in the API surface
> says so.

`[M]` Two more template facts that will bite an agent harness, both zero-install checkable:

- `chat_template.jinja:9` hardcodes a default system message ("You are a helpful,
  conversationally-fluent assistant made by Poolsideâ€¦") that is injected whenever the caller
  supplies none. `:8` documents the opt-out: pass a system message with **empty content**.
- Do that while `enable_thinking=True` and `:16` still fires (`has_sys or tools or
  enable_thinking`), so you get a literally empty `<system></system>` block. Harmless, but
  it is a real difference in the token stream and therefore in the prefix hash.

---

## 4. Why this matters for Proteus and Mnemosyne

**4.1 Laguna-XS is Mnemosyne's first real integration test, and the spec is already
sufficient.** `kv-cache-mechanics.md` Â§4.1 argued that Mnemosyne's cost model should consume
`list[LayerCacheSpec]` and nothing else â€” never a model object â€” because the boundary rule
forbids importing Proteus and because per-layer specs are not optional. This module supplies
the concrete instance: forty specs, ten with `window=None` and `n_q=48`, thirty with
`window=512` and `n_q=64`, all with `n_kv=8, d_h=128, dtype=bf16`. That list is derivable
from `config.json` with `json.load` and no torch. **Build the spec extractor before the
weights land**, point it at Laguna-XS, Laguna-S and gpt-oss-20b, and check it reproduces
40,960 / 62,914,560 / 163,840 for XS. It is the cheapest possible separability test and it
runs today.

**4.2 The 11.89Ã— capacity-to-traffic ratio is the argument for the whole hardware strategy,
and it is testable.** `ASSUMPTIONS.md â†’ z13-is-right-instrument` is currently **untested,
status downgraded** â€” the load-bearing claim rests on one un-peer-reviewed GitHub issue, and
two other numbers from that issue family have already failed locally. Laguna-XS gives a
clean local test of the *capacity* half: a 33B model whose per-token traffic is 5.24 GiB
cannot run on a 20 GB card at all, and runs here with room for a 10 GiB KV cache. That is a
qualitative difference, not a benchmark. It does not settle the MFU half of the claim, which
still needs a known-good recipe at the Hardware Validation Gate.

**4.3 The unfused expert layout is a Mnemosyne research object, not an implementation
detail.** `[M]` 29,952 individually-addressable 2 MiB expert tensors, of which 936 are read
per decode step. That is a *content-addressed working set with a learned access pattern* â€”
the router's top-k is a predictor, and the correction bias
(`architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:185`) is a load-balancing signal that
already exists. Everything Mnemosyne knows about KV tiering â€” hot/warm/cold, admission,
prefetch â€” has an analogue here, with two differences that make it *easier*: an expert
weight is **immutable** and **exactly recomputable-free** (it can be re-read from disk, unlike
a KV entry which cannot be recomputed without a full prefill). Expert offload is the one
tiering problem in this system with a real backing store. It is out of scope for the
mnemosyne-core milestone and it should be in `BACKLOG.md`.

**4.4 The gating divergence in Â§2.5(a) is a Proteus config-surface requirement.** Every
ablation axis is a config field, and the config surface *is* the experimental surface. Two
shipped implementations disagree on what a boolean means. The lesson for Proteus's typed
config object is not "be careful" â€” it is **forbid the boolean**. `gating: Literal["off",
"per-head", "per-element"]`, no `True`, no default that a reader has to look up. Same for
`layer_types`: require the explicit list, never derive it from a period, because
`cache_utils.py:1645`'s fallback ladder will happily make *every* layer sliding if a bare
`sliding_window` field is present without `layer_types`, with no error.

**4.5 The prefill hazard in Â§3.5(1) is a Hardware Validation Gate item with a number
attached.** T = 16,384 for a 64-head layer at batch 1. The gate's numerics suite must run
both with and without `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` before that flag can become
a default, because it is an experimental kernel path and therefore a numerics change on a
machine where `bf16-numerics-unproven` is still open. Until the gate closes, **no result from
a Laguna-XS run on this machine is evidence**; it is instrument shakedown, and it should be
labelled as such in the notebook.

---

## 5. Read the code

All paths relative to `research/reference/`. Clones are gitignored; `scripts/fetch_reference.sh`
materialises them. Line numbers are pinned to the revisions in `PROVENANCE.md`
(`laguna-xs` @ `205dc65dd4bd`, `transformers` @ `b6d5084fb4a5`, `llama-cpp-laguna` @
`04b2b72cb540`, `vllm` @ `0934b267906f`).

### 5.1 The artifact itself â€” read this before any code

| Where | What to look at, and why |
|---|---|
| `models/laguna-xs/config.json:59` | `"layer_types": [` â€” 40 strings. The whole hybrid mechanism is this list. Count them yourself: 10 `full_attention` at indices 0,4,â€¦,36. |
| `models/laguna-xs/config.json:41` | `"sliding_window": 512` â€” one scalar shared by all thirty windowed layers. There is no per-layer window. |
| `models/laguna-xs/config.json:187` | `"num_attention_heads_per_layer": [` â€” 48/64 by layer type, contradicting `num_attention_heads: 48` on line 14 for 30 of 40 layers. `num_key_value_heads` has no such list, which is why the KV product is unaffected. |
| `models/laguna-xs/config.json:42` | `"rope_parameters"` â€” **two** rope configs, keyed by layer type. YaRN Î¸=500000 with `partial_rotary_factor: 0.5` on global layers; plain Î¸=10000 over all 128 dims on sliding ones. This is why you cannot widen the windows. |
| `models/laguna-xs/model.safetensors.index.json:3` | `"total_size": 66885234176` â€” the number Â§3.2 closes against. Below it, `weight_map` names every tensor; grep `experts\.` and count. |
| `models/laguna-xs/model-00001-of-00014.safetensors:1` | `version https://git-lfs.github.com/spec/v1` â€” the three-line pointer stub. Line 3 gives the real size. This is what "we hold config only" looks like on disk. |
| `models/laguna-xs/generation_config.json:13` | `"speculative_config"` â€” the checkpoint ships a **default speculator**: DFlash, 15 tokens. `[C]` (2602.06036). Note the disagreement with `README.md:158`, which says 7. Two numbers, same repo. |
| `models/laguna-xs/README.md:33` | "KV cache in FP8" as a headline feature â€” and Â§3.4 on why that is a property of the serving stack, not of the checkpoint. |

### 5.2 The two implementations, side by side

Open both files. This is the highest-value read in the module.

| Where | What to look at, and why |
|---|---|
| `models/laguna-xs/modeling_laguna.py:375` | `self.is_sliding = layer_types[layer_idx] == "sliding_attention"` â€” a list lookup at construction time. Every hybrid-ratio question this lab asks is a question about that list. |
| `models/laguna-xs/modeling_laguna.py:479` | `per_layer_heads = getattr(config, "num_attention_heads_per_layer", None)` â€” the decoder layer, not the attention module, resolves the head count and passes it down. `getattr` with a `None` default: an older config silently falls back to the uniform count. |
| `models/laguna-xs/modeling_laguna.py:434` | `past_key_values.update(key_states, value_states, self.layer_idx)` â€” the one line where bytes enter the cache. Note what already happened: QK-norm at `:427-428`, RoPE at `:431`. **A cached key is `RoPE(RMSNorm(k))`**, doubly transformed, which constrains every requantisation or re-positioning scheme downstream. |
| `models/laguna-xs/modeling_laguna.py:666` and `:668` | Two masks built per forward, one per layer type, then `:686` dispatches `causal_mask_mapping[decoder_layer.attention_type]`. The hybrid is a dictionary lookup at three separate places in the stack. |
| `models/laguna-xs/modeling_laguna.py:607` | `if getattr(config, "swa_rope_parameters", None) is not None:` â€” and `else: self.swa_rotary_emb = None` at `:616`, which makes the sliding layers reuse the global rope. Combined with `configuration_laguna.py:192`, this is a two-file, silent-failure path. |
| `models/laguna-xs/configuration_laguna.py:189-193` | The derivation comment that names the failure outright. Read the comment, not just the code. |
| `models/laguna-xs/configuration_laguna.py:37-42` vs `architecture/transformers/src/transformers/models/laguna/configuration_laguna.py:33-36` | **The contradiction.** Two docstrings for the same field, saying opposite things about `True`. Then `models/laguna-xs/modeling_laguna.py:394` vs `architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:370` for the code that implements each. |
| `models/laguna-xs/modeling_laguna.py:400` | `if self.is_sliding and getattr(config, "swa_attention_sink_enabled", False):` â€” learnable attention sinks `[C]` (2309.17453) exist in the code and are **off** for this checkpoint, because `config.json` omits the key. A capability present in the implementation and absent from the artifact. |
| `models/laguna-xs/modeling_laguna.py:459` | `gate = F.softplus(self.g_proj(hidden_states).float())` â€” the output gate, computed in fp32 and cast back. One scalar per head here; per channel under the other implementation's default. |

### 5.3 The same model in C++, where the tiering is explicit

| Where | What to look at, and why |
|---|---|
| `architecture/llama-cpp-laguna/src/models/laguna.cpp:41` | `set_swa_pattern` â€” period 4, full at `il%4==0`. A striping function where HuggingFace has a map (Â§2.5c). Also: if the `sliding_window` key is absent the whole hybrid path is skipped and you get an all-global model, silently. |
| `architecture/llama-cpp-laguna/src/llama-kv-cache-iswa.cpp:73` | `size_swa` â€” two physically separate cache allocations, the small one sized `min(full, n_swaÂ·n_seq + n_ubatch)` padded to 256. Â§3.4's `L_w Â· w` term, as an allocator call. |
| `architecture/llama-cpp-laguna/src/llama-graph.cpp:2891` | `is_swa` â€” per-layer dispatch selecting which of the two cache contexts receives this layer's writes and supplies its mask. |
| `architecture/llama-cpp-laguna/src/models/laguna.cpp:184` | `n_rot_l` â€” where the per-layer RoPE divergence is implemented: SWA layers get plain RoPE with YaRN's `ext_factor`, `beta_fast`, `beta_slow` all forced to zero. |
| `architecture/llama-cpp-laguna/src/llama-kv-cache.cpp:319` | `attn_rot_k` â€” a Hadamard rotation applied to K/V before a **quantised** cache store, gated by a runtime heuristic (quantised type AND `head_dim % 64 == 0`), disableable by an undocumented `LLAMA_ATTN_ROT_DISABLE`. Nothing in the model config mentions it. If you benchmark quantised KV under llama.cpp you are benchmarking this too. |
| `architecture/llama-cpp-laguna/src/models/laguna.cpp:154` | The whole forward graph in ~200 readable lines â€” QKV, QK-norm, per-layer RoPE, softplus gate, sigmoid-routed MoE with an always-on shared expert. The best single read for "what does this model actually compute." |

### 5.4 What a real server would do with it, which we cannot run

| Where | What to look at, and why |
|---|---|
| `memory/vllm/vllm/v1/kv_cache_interface.py:227` | `class FullAttentionSpec` â€” **read the docstring.** With the hybrid allocator disabled, sliding-window layers are "regarded as full attention in KV cache manager (blocks are allocated for all tokens), while computed as sliding window attention in model runner." In plain terms: **one feature flag off and Laguna's 3.98Ã— capacity saving silently does not happen.** Model unchanged, config unchanged, residency 4Ã— higher. |
| `memory/vllm/vllm/v1/kv_cache_interface.py:204` | `real_page_size_bytes` â€” `2 * block_size * num_kv_heads * head_dim * dtype_size`. Our `c`, times block size, in production code. |
| `memory/vllm/vllm/v1/core/block_pool.py:647` | `get_new_blocks` â€” on failure there is no fault and no demotion; the request is preempted and its prefill re-run. There is no miss path anywhere in a KV cache. |
| `architecture/transformers/src/transformers/cache_utils.py:1645` | `get_layer_types_and_kwargs` â€” the fallback ladder when `layer_types` is absent, and `:1653`, where a bare `sliding_window` makes **every** layer sliding. The mirror image of llama.cpp's failure at `laguna.cpp:41`: one defaults to all-global, the other to all-sliding, from the same missing key. |

### 5.5 The tokenizer and template, as data

| Where | What to look at, and why |
|---|---|
| `models/laguna-xs/tokenizer.json:638-663` | The pre-tokenizer is a **Sequence of three**: a `Split` on runs of newlines with `MergedWithNext` behaviour (`:644`), then the familiar GPT-style contraction/word/number/whitespace regex (`:652`), then `ByteLevel` with `use_regex: false`. That first split is a coding-model design choice â€” it keeps a newline run attached to the line that follows it, which is what makes indentation tokenise stably. |
| `models/laguna-xs/tokenizer.json:732` | `"byte_fallback": false` on a byte-level BPE â€” the ByteLevel pre-tokenizer already guarantees coverage, so there is nothing to fall back to. Contrast with a SentencePiece tokenizer where this flag is load-bearing. |
| `models/laguna-xs/tokenizer.json:665-717` | `TemplateProcessing` prepending `ã€ˆ|EOS|ã€‰`. Read it next to `chat_template.jinja:3`, which does the same thing. Â§3.6. |
| `models/laguna-xs/tokenizer_config.json:564` and `:567` | `bos_token` and `eos_token` are the **same string**. |
| `models/laguna-xs/tokenizer_config.json:515-562` | The six chat tags that are single tokens, all with `"special": false` (`:521`). Everything else in the template is ordinary text. |
| `models/laguna-xs/tokenizer_config.json:570` | `model_max_length: 1000000000000000019884624838656` â€” the sentinel meaning "unset". The real limit is `max_position_embeddings: 262144` in `config.json:17`, and nothing cross-checks them. |
| `models/laguna-xs/chat_template.jinja:44` and `:76` | `{%- generation -%}` / `{%- endgeneration -%}` â€” not standard Jinja. It is a transformers extension marking the assistant span so the *same template* can produce training-time label masks. Stock `jinja2` will not parse it; Exercise B strips it and explains why that is safe for rendering and not safe for training. |
| `models/laguna-xs/chat_template.jinja:55` vs `:57` | Thinking on emits `<think>â€¦</think>`; thinking off emits a bare `</think>` with no opener. Deliberate, and it looks like a bug in a diff. |

---

## 6. Exercises

**All three run today, with no weights.** They need `python`, the standard library, and (for
C) `torch`. `jinja2` and `numpy` are already in `C:\venvs\lab`; `transformers` is **not**, and
none of these exercises requires it.

I did not run them. Exercises A and B produce **exact integers** that I derived by hand from
the artifacts and stated in Â§3 â€” if your run disagrees with the expected values below, one of
us is wrong and the artifact settles it. Exercise C produces a measured slope whose *expected*
value is exact arithmetic; its wall-clock and allocator-overhead numbers are genuinely
unknown and are the part worth writing down.

Activate first, dot-sourced so the variables survive:

```powershell
. .\scripts\activate-lab.ps1
```

**Standing hardware caveats** (`ASSUMPTIONS.md`): keep every single tensor under 31 GiB â€”
â‰¥32 GiB **hangs silently at 0% CPU**; bf16 numerics on gfx1151 are unproven, so any accuracy
claim is provisional while capacity and timing claims are not;
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` is deliberately **off** and these exercises assume
it is off. The Hardware Validation Gate has not run.

Write scratch scripts under `notebook/`. Exercise A is a rig candidate â€” it is the
`LayerCacheSpec` extractor from Â§4.1 and should migrate into `mnemosyne` with tests on reuse.

---

### Exercise A â€” reconstruct the model from `config.json` and close the books against the index

**Goal:** derive the parameter count, the active-parameter count, and the KV geometry from
the spec alone, and check the first against `total_size`. This is the strongest verification
available without weights, and it is the Mnemosyne spec extractor in embryo.

**Hardware:** none. Pure stdlib. **CPU fallback:** n/a â€” there is nothing else it could run
on. **Runtime:** under one second.

```python
"""Laguna-XS 2.1: everything the spec determines, with no weights.
Reads research/reference/models/laguna-xs/{config.json,model.safetensors.index.json}."""
import json, pathlib

ROOT = pathlib.Path(r"C:\projects\School\chiron\research\reference\models\laguna-xs")
cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
idx = json.loads((ROOT / "model.safetensors.index.json").read_text(encoding="utf-8"))

d      = cfg["hidden_size"]
d_h    = cfg["head_dim"]
n_kv   = cfg["num_key_value_heads"]
L      = cfg["num_hidden_layers"]
types  = cfg["layer_types"]
heads  = cfg.get("num_attention_heads_per_layer") or [cfg["num_attention_heads"]] * L
w      = cfg["sliding_window"]
V      = cfg["vocab_size"]
E, k   = cfg["num_experts"], cfg["num_experts_per_tok"]
d_e    = cfg["moe_intermediate_size"]
d_s    = cfg["shared_expert_intermediate_size"]
d_ff   = cfg["intermediate_size"]
dense  = set(cfg["mlp_only_layers"])
assert len(types) == len(heads) == L, "per-layer lists must match num_hidden_layers"

def attn_params(n_q):
    return (d * n_q * d_h          # q_proj
          + d * n_kv * d_h         # k_proj
          + d * n_kv * d_h         # v_proj
          + n_q * d_h * d          # o_proj
          + d * n_q                # g_proj, gating == "per-head"
          + 2 * d_h)               # q_norm + k_norm

moe_layer   = E * 3 * d * d_e + 3 * d * d_s + d * E + E   # experts, shared, router, bias
moe_active  = k * 3 * d * d_e + 3 * d * d_s + d * E + E
dense_layer = 3 * d * d_ff

total = sum(attn_params(h) for h in heads)
total += sum(dense_layer if i in dense else moe_layer for i in range(L))
total += 2 * V * d                      # embed_tokens + lm_head (tie_word_embeddings false)
total += L * 2 * d + d                  # two norms per layer, plus the final norm

active = sum(attn_params(h) for h in heads)
active += sum(dense_layer if i in dense else moe_active for i in range(L))
active += V * d                         # lm_head; embed_tokens is a one-row gather
active += L * 2 * d + d

reported = idx["metadata"]["total_size"]
c      = 2 * n_kv * d_h * 2             # bytes per token per layer, bf16
L_g    = types.count("full_attention")
L_w    = types.count("sliding_attention")

print(f"layers                 {L}  ({L_g} full, {L_w} sliding, window {w})")
print(f"query heads            {sorted(set(heads))}   kv heads {n_kv}   G = {sorted({h//n_kv for h in heads})}")
print(f"params derived         {total:,}")
print(f"params from index      {reported // 2:,}   (total_size {reported:,} B / 2)")
print(f"MATCH                  {total * 2 == reported}")
print(f"active per token       {active:,}   ({active/total:.2%} of the model)")
print(f"capacity:traffic       {total/active:.2f}x")
print(f"weights bf16           {reported / 2**30:.4f} GiB")
print(f"KV bytes/token/layer   {c:,}")
print(f"KV slope (growing)     {c * L_g:,} B/token       nominal all-global {c * L:,}")
print(f"KV fixed term          {c * L_w * w:,} B")
for T in (512, 1536, 32768, 131072, cfg["max_position_embeddings"]):
    r = c * L_g * T + c * L_w * min(T, w)
    print(f"  R({T:>7,}) = {r/2**30:8.4f} GiB    all-global {c*L*T/2**30:8.4f} GiB"
          f"    saving {c*L*T/r:.2f}x")
print(f"byte-parity T*         {L_w * w // L_g:,} tokens")
```

**Predictions, stated before you run.**
1. `MATCH` is `True`. If it is not, your decomposition is wrong â€” the artifact is not.
2. The saving ratio rises monotonically toward `L/L_g = 4.00` and never reaches it.
3. `active/total` is under 10%.

**Expected values** (derived by hand in Â§3.2â€“Â§3.4; these are the checkable numbers):

| Quantity | Value |
|---|---|
| params derived | **33,442,617,088** |
| params from index | **33,442,617,088** â€” `MATCH True` |
| active per token | **2,811,614,976** (8.41%) |
| capacity:traffic | **11.89Ã—** |
| weights bf16 | **62.2917 GiB** |
| KV slope | **40,960 B/token** (nominal all-global 163,840) |
| KV fixed term | **62,914,560 B = 60 MiB** |
| R(262,144) | **10.0586 GiB** vs 40.0000 GiB all-global, saving 3.98Ã— |
| byte-parity T* | **1,536 tokens** |

**Deliverables â€” three numbers and one decision.**
1. The `MATCH` boolean, plus the derived and reported parameter counts side by side.
2. The KV slope and fixed term. Write them into your notes as the two numbers that size
   every long-context experiment on this model.
3. Re-point `ROOT` at `models/laguna-s` and run it again. Report the same table for the 118B
   model. `[M]` `ASSUMPTIONS.md â†’ kv-per-token-laguna` says its nominal is 192 KiB/token;
   check that you reproduce it, and check whether its parameter count also closes to the
   byte. **Then decide:** which of the two is your reference for Mnemosyne's first cost-model
   test, and why. (Hint: only one of them can be run.)

**What a falsification would mean.** If `MATCH` is `False` by a small amount, look first at
`tie_word_embeddings`, then at whether `e_score_correction_bias` is per-expert, then at the
QK-norm shape. If it is off by a factor near 2, you have an fp32 tensor somewhere and the
"everything is bf16" conclusion in Â§3.2 is wrong â€” which is a genuinely interesting finding
and worth a notebook entry.

---

### Exercise B â€” render the chat template and do tokenizer forensics, with zero installs

**Goal:** establish the prompt contract before you can run the model, and find the three
behaviours in it that will surprise an agent harness.

**Hardware:** none. Needs `jinja2`, which is already in the lab venv. **CPU fallback:** n/a.
**Runtime:** under two seconds.

```python
"""Laguna-XS 2.1 prompt contract, from chat_template.jinja + tokenizer_config.json.
No transformers, no tokenizers, no weights."""
import json, pathlib, re
from jinja2 import Environment

ROOT = pathlib.Path(r"C:\projects\School\chiron\research\reference\models\laguna-xs")
src  = (ROOT / "chat_template.jinja").read_text(encoding="utf-8")
tcfg = json.loads((ROOT / "tokenizer_config.json").read_text(encoding="utf-8"))
tok  = json.loads((ROOT / "tokenizer.json").read_text(encoding="utf-8"))

# {% generation %}/{% endgeneration %} is a transformers Jinja extension that marks the
# assistant span so the SAME template can emit training-time label masks. Stock jinja2
# cannot parse it. Stripping it is safe for RENDERING and destroys the mask, so never
# reuse this stripped template for SFT data generation.
src = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", src)

env = Environment(trim_blocks=False, lstrip_blocks=False)
env.filters["tojson"] = lambda v, **kw: json.dumps(v, **{"ensure_ascii": True, **kw})
tpl = env.from_string(src)

msgs = [{"role": "user", "content": "Write a retry wrapper with exponential backoff."}]

on  = tpl.render(messages=msgs, add_generation_prompt=True, enable_thinking=True)
off = tpl.render(messages=msgs, add_generation_prompt=True, enable_thinking=False)
nosys_on = tpl.render(messages=[{"role": "system", "content": ""}] + msgs,
                      add_generation_prompt=True, enable_thinking=True)

print("--- thinking ON ---");  print(repr(on))
print("--- thinking OFF ---"); print(repr(off))
print("--- empty system + thinking ON ---"); print(repr(nosys_on))
print(f"len(on)={len(on)}  len(off)={len(off)}  delta_chars={len(on)-len(off)}")
print(f"prefix identical up to char {next(i for i,(a,b) in enumerate(zip(on,off)) if a!=b)}")

added = {int(i): v["content"] for i, v in tcfg["added_tokens_decoder"].items()}
chat_tags = {i: s for i, s in added.items() if s.startswith("<") and not s.startswith("ã€ˆ")}
print("\nchat tags that are single tokens:", dict(sorted(chat_tags.items())))
print("their 'special' flags:",
      {int(i): v["special"] for i, v in tcfg["added_tokens_decoder"].items()
       if int(i) in chat_tags})

vocab = tok["model"]["vocab"]
for probe in ["<system>", "<user>", "<tool_response>", "<arg_key>", "<think>", "</assistant>"]:
    print(f"  in BPE vocab: {probe:<16} {probe in vocab}")

print("\nbos_token", tcfg["bos_token"], "| eos_token", tcfg["eos_token"],
      "| same:", tcfg["bos_token"] == tcfg["eos_token"])
pp = tok["post_processor"]
print("post_processor type:", pp["type"],
      "| prepends:", [x["SpecialToken"]["id"] for x in pp["single"] if "SpecialToken" in x])
print("template line 3 emits:", repr(src.splitlines()[2].strip()))
print("reserved ã€ˆ|SPECIAL_n|ã€‰ slots:",
      sum(1 for v in added.values() if v.startswith("ã€ˆ|SPECIAL_")))
print("vocab size (BPE):", len(vocab), "| config vocab_size:",
      json.loads((ROOT / "config.json").read_text(encoding="utf-8"))["vocab_size"])
```

**Predictions, stated before you run.**
1. `delta_chars == 0` â€” thinking on and off produce prompts of **identical length**; they
   differ in exactly one tag at the very end.
2. `<think>`, `</think>`, `<assistant>`, `</assistant>`, `<tool_call>`, `</tool_call>` are
   the only chat tags in the added-token table, and all six have `special: false`.
3. `<system>`, `<user>`, `<tool_response>`, `<arg_key>` are **not** in the BPE vocab at all.
4. `bos_token == eos_token`, and the post-processor prepends the same token the template
   already emitted.
5. The empty-system render still emits a `<system></system>` block, because
   `chat_template.jinja:16` is satisfied by `enable_thinking` alone.

**Deliverables â€” three numbers and one hazard write-up.**
1. `delta_chars`, and the character index at which the two renders diverge. Explain in one
   sentence why a serving stack can toggle thinking on the *last* turn without invalidating
   a prefix cache, and why toggling it with **preserved thinking** in the history is a full
   invalidation. (Â§3.6.)
2. The count of reserved `ã€ˆ|SPECIAL_n|ã€‰` slots (expected: **46**) and the BPE vocab size
   against `config.vocab_size` (100,352). Any gap is padding to a multiple that suits the
   matmul, not a bug â€” report the gap and say what multiple it lands on.
3. The `special: false` finding, written up as a hazard with the exact test that would
   confirm it once `transformers` is installed: build the id list `[23, 19, <some text ids>,
   24]` by hand, `decode(..., skip_special_tokens=True)`, and check whether `</think>` and
   `<assistant>` survive. **Predict the answer before you run it.** The model card's own
   example (`README.md:217`) depends on this.

**What a falsification would mean.** If `delta_chars != 0`, re-read
`chat_template.jinja:86-92`: you have probably left `add_generation_prompt` false, in which
case the thinking flag only affects historical assistant turns and the delta is real. If
`<user>` *is* in the vocab, the tokenizer was updated after revision `205dc65dd4bd` and every
prefix-length calculation in this module needs re-deriving.

---

### Exercise C â€” make the sliding window visible in allocated bytes

**Goal:** the module's central claim, measured. Allocate Laguna-XS's exact per-layer cache
geometry at a sweep of context lengths and confirm the residency slope is **40,960 B/token**,
not the 163,840 the top-level config implies. This is the check that catches a dropped
`layer_types` list â€” the failure mode that produces a working model with a 4Ã— memory bill and
no error.

**Hardware:** one gfx1151 GPU. **CPU fallback:** set `DEV = "cpu"`; the analytic sum is
identical and the allocator column is simply absent, so the deliverable survives intact.
**Runtime:** `[A]` a few seconds to a couple of minutes â€” the largest sweep point allocates
10.06 GiB in 80 tensors and the dominant cost is the allocation itself, not compute. Unknown
precisely; report yours.

**Footprint check first, because this is a capacity lab:** the largest single tensor is a
global layer's K at `T = 262,144`: `8 Ã— 262,144 Ã— 128 Ã— 2 B = 512 MiB`. Total at that point
is 10.06 GiB. Both far inside the `[M]` â‰¥62 GiB fast tier and far below the 31 GiB
per-tensor hazard.

```python
"""KV residency slope for Laguna-XS 2.1's exact layer schedule, read from config.json.
Allocates the real per-layer tensors; no weights, no model, no transformers."""
import json, pathlib, torch

ROOT = pathlib.Path(r"C:\projects\School\chiron\research\reference\models\laguna-xs")
cfg  = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
DEV  = "cuda" if torch.cuda.is_available() else "cpu"
DT   = torch.bfloat16

types = cfg["layer_types"]
n_kv, d_h, w = cfg["num_key_value_heads"], cfg["head_dim"], cfg["sliding_window"]
c = 2 * n_kv * d_h * DT.itemsize
L_g = types.count("full_attention")
L_w = types.count("sliding_attention")
print(f"{L_g} full + {L_w} sliding, window {w}, c = {c} B/token/layer, device {DEV}")

def allocate(T, honour_window: bool):
    """One K and one V per layer, at that layer's true residency."""
    kv, analytic = [], 0
    for t in types:
        n = min(T, w) if (honour_window and t == "sliding_attention") else T
        for _ in range(2):
            x = torch.empty(n_kv, n, d_h, dtype=DT, device=DEV)
            analytic += x.numel() * x.element_size()
            kv.append(x)
    return kv, analytic

print(f"\n{'T':>9} {'hybrid GiB':>11} {'all-global GiB':>15} {'saving':>7} {'alloc GiB':>10}")
rows = []
for T in [512, 1536, 4096, 32768, 131072, cfg["max_position_embeddings"]]:
    if DEV == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    kv, hybrid = allocate(T, honour_window=True)
    alloc = torch.cuda.memory_allocated() if DEV == "cuda" else float("nan")
    del kv
    if DEV == "cuda":
        torch.cuda.empty_cache()
    kv2, naive = allocate(T, honour_window=False)   # the dropped-layer_types control
    del kv2
    if DEV == "cuda":
        torch.cuda.empty_cache()
    rows.append((T, hybrid))
    print(f"{T:>9,} {hybrid/2**30:>11.4f} {naive/2**30:>15.4f} "
          f"{naive/hybrid:>7.2f} {alloc/2**30 if alloc == alloc else float('nan'):>10.4f}")

(T1, R1), (T2, R2) = rows[-2], rows[-1]
slope = (R2 - R1) / (T2 - T1)
print(f"\nmeasured slope    {slope:,.1f} B/token")
print(f"predicted slope   {c * L_g:,} B/token   (all-global would be {c * len(types):,})")
print(f"intercept         {R2 - slope * T2:,.0f} B   predicted {c * L_w * w:,}")
```

**Predictions, stated before you run.**
1. `measured slope` is **exactly 40,960.0** B/token â€” this is arithmetic over integers, not a
   fit, so any deviation is a bug in the harness or a changed config.
2. `intercept` is **exactly 62,914,560** B.
3. The saving column rises from **1.00Ã— at T = 512** through **2.00Ã— at T = 1,536** to
   **3.98Ã— at T = 262,144**, and never reaches 4.00.
4. `alloc GiB` exceeds the analytic figure slightly â€” allocator rounding across 80 tensors.
   **How much is genuinely unknown**; it is the one thing in this exercise I cannot predict,
   and it is the number worth writing down, because it is the per-tensor overhead that a
   paged Mnemosyne allocator would be competing against.

**Deliverables â€” three numbers and one plot.**
1. The measured slope and intercept, against the predicted 40,960 / 62,914,560. Exact match
   or an explanation.
2. The allocator overhead: `alloc âˆ’ analytic`, in bytes and as a percentage, at
   `T = 262,144`. This is a number this lab does not have.
3. Plot hybrid and all-global residency against `T` on a log-x axis and mark `T* = 1,536`.
   Then mark the `[M]` â‰¥62 GiB fast tier as a horizontal line and read off where each curve
   crosses it. **Neither does**, within the model's 262k context â€” which is the quantitative
   statement of "this machine can hold Laguna-XS's full-context KV cache four times over,
   and cannot hold its bf16 weights once."

**Extension, once the weights land â€” the protocol, not an exercise.** In order, each step
gated on the previous one passing:

```
1. Load with device_map="auto", dtype=torch.bfloat16, low_cpu_mem_usage=True.
   Watch host RAM: 31.6 GB available, shards are 4.77 GiB. A path that spikes past
   ~10 GB host is materialising something it should be streaming.
2. Assert the schedule survived the load:
     [ (i, m.self_attn.sliding_window, m.self_attn.num_heads)
       for i, m in enumerate(model.model.layers) ]
   Expect sliding_window None on exactly 10 layers and 512 on 30; num_heads 48 on the
   ten Nones and 64 on the thirty 512s. If num_heads is uniformly 48, the per-layer
   head list was dropped -- see modeling_laguna.py:479.
3. Assert the two rope modules exist and differ:
     model.model.rotary_emb is not model.model.swa_rotary_emb  and  the latter is not None.
   If swa_rotary_emb is None, configuration_laguna.py:192 did not fire and thirty layers
   are running the wrong positional encoding. Silent. (Section 2.5b.)
4. PREFILL LIMIT. Do not exceed T = 4096 in bf16 under transformers until you have
   measured the score-matrix path on this build. Section 3.5(1): 64 * T^2 * 2 bytes per
   windowed layer, which reaches the [M] 32 GiB silent-hang threshold at exactly
   T = 16,384. Start at T = 512 and double, watching torch.cuda.max_memory_allocated().
5. Decode with a DynamicCache and print, every 64 steps,
     [ cache.layers[i].keys.shape[-2] for i in range(40) ]
   Global layers grow by 1 per step; sliding layers pin. Expect 511, not 512 --
   cache_utils.py:240 stores w-1 and returns w. The retained tensor and the attended
   tensor are different objects and any instrumentation must say which it counts.
6. Only now measure the residency slope on the real model and compare against the
   40,960 B/token this exercise established with no weights at all.
```

Step 6 is the point of the whole module: **you already know the answer before the download
finishes, so the measurement is a test of the stack rather than a discovery about the
model.** That is the difference between instrumentation and hoping.

---

## 7. Self-check

1. `config.json` says `num_attention_heads: 48`. You need the KV cache size at 128k context.
   Does the 48 matter? Does it matter for anything else? Be specific about which quantity
   each head count enters.
2. You have a fast tier measured good to â‰¥62 GiB and a bf16 checkpoint of 62.29 GiB. Someone
   proposes running it anyway, arguing that the fast tier is a floor and not an edge. Give
   the two strongest reasons the run will still fail, in order of how early it fails.
3. The residency slope for Laguna-XS is 40,960 B/token. You instrument a run and measure
   163,840. Name three distinct mechanisms that produce exactly that number, in three
   different pieces of software, and say what single check distinguishes them.
4. A colleague reports that turning off thinking "saves a lot of prompt tokens." Is that
   right? Under what circumstance does the thinking flag change more than one token, and
   what does that do to a prefix cache?
5. Laguna-XS holds 33.44B parameters and reads 2.81B per token â€” an 11.89Ã— ratio. Your
   instinct from database work says to improve locality by batching probes. Explain
   precisely which term batching improves and which term it makes worse, and why that is the
   opposite of the hash-join intuition.
6. Two implementations of Laguna ship in this repository and both default `gating=True`.
   What is the observable difference, why does Laguna-XS 2.1 not exhibit it, and what is the
   one-line change to Proteus's config surface that makes the class of bug impossible?

---

## 8. What is still unsolved

1. **Nothing in this module has been run on a loaded model.** That is the single largest gap
   and it is a download away. The slope prediction (40,960 B/token), the prefill hang
   threshold (T = 16,384), the decode floor (â‰ˆ28.4 tok/s at 32k) and the double-BOS hazard are
   all derived. Every one of them is falsifiable by the step-1-to-6 protocol in Exercise C.
2. **The Q4_K_M and FP8 file sizes are arithmetic, not reads.** Fetching two index files
   costs kilobytes and would upgrade Â§3.5's decision table from `[A]` to `[M]`. Do that
   before the 62 GiB download, not after.
3. **The MoE gather bandwidth is completely uncharacterised.** `[M]` We have 199.9 GB/s for a
   contiguous copy and ~150 GB/s for a decode-shaped attention read. We have nothing for a
   936-way gather over 2 MiB blocks, which is what a Laguna decode step actually is, and it
   is the term that dominates the step. This is the highest-value missing measurement in the
   module and it does **not** need weights â€” random tensors of the right shape and a
   router-shaped index would answer it in an afternoon. It is also a direct test of
   `ASSUMPTIONS.md â†’ z13-is-right-instrument`.
4. **Whether the unfused expert layout is a serving liability or an offload opportunity is
   genuinely open.** 29,952 addressable 2 MiB tensors is either a per-tensor-overhead
   disaster at load time or the natural granularity for an expert cache with a real backing
   store. Nobody in the surveyed literature treats expert weights and KV entries as tiers of
   one memory system, and the asymmetry â€” expert weights are immutable and re-readable, KV
   entries are neither â€” suggests they should not be managed by the same policy. Unresolved
   and, I think, a genuine research opening.
5. **Contested: does a quantised KV cache pay on a unified-memory machine?** `[C]`
   2605.05699 (May 2026) reports an int4 KV cache *outrunning* fp16 on Apple Silicon,
   precisely because the dequantise hides behind the bytes saved. `[M]` Our own measurement
   went the other way â€” fp8 KV was 2.9â€“3.1Ã— **slower** under PyTorch on gfx1151
   (`kv-cache-mechanics.md` Exercise C) â€” but that path lacks `torch._scaled_mm` and
   materialises a bf16 temporary, which is not the path the Apple result used. The two
   findings are not in contradiction; they are measurements of different implementations.
   Whether llama.cpp's `-ctk q8_0` path on this hardware behaves like the Apple result or
   like ours is unmeasured, and Laguna-XS under llama.cpp is the natural test case.
6. **The `swa_attention_sink_enabled` code path is dead in this checkpoint and we do not
   know why.** `[C]` Attention sinks (2309.17453) are the standard fix for the streaming
   failure mode of a windowed layer, the implementation is present
   (`models/laguna-xs/modeling_laguna.py:400-401`), and the config simply omits the key.
   Either Poolside found them unnecessary given ten global layers, or the feature postdates
   the checkpoint. This is answerable from the technical report (`README.md:29`) and is not
   answered here.
7. **The DFlash speculator's token count is stated twice, differently.** `[M]`
   `generation_config.json:17` says `num_speculative_tokens: 15`; `README.md:158` says "up to
   7 tokens per step". `[A]` Probably the difference between the draft block size and the
   recommended acceptance depth, but that is a guess. It matters because speculation is a
   KV-capacity consumer â€” `research/notes/inference-and-quantization.md` flags that
   speculation and batch size compete for the same bytes â€” and a 2Ã— error in the reserved
   lookahead is a 2Ã— error in a capacity budget. Unresolved.
8. **We cannot observe this model under a real serving stack, and that is a permanent gap on
   this machine, not a to-do.** Â§5.4's `FullAttentionSpec` docstring says a single feature
   flag silently costs Laguna 4Ã— its capacity saving. That is exactly the class of finding
   this lab exists to catch, and we cannot catch it here, because vLLM and SGLang have no
   native-Windows build path `[A]` (high confidence; the cheapest disproof is a successful
   `pip install vllm` on this box) and distributed collectives are incomplete on gfx1151 `[M]`
   regardless. Reading the code is not the same as running it, and this module does not
   pretend otherwise.

---

## Answers to the self-check

**1.** The 48 does not matter for KV size and matters for almost everything else. KV bytes are
`2Â·LÂ·n_kvÂ·d_hÂ·b` â€” **query heads do not appear in the product**, and `num_key_value_heads` is
a uniform 8 with no per-layer override, so the KV geometry is exact from the top-level config:
4096 B per token per layer, 40,960 B/token growing. What the 48 *is* wrong about is the GQA
group size `G = n_q/n_kv`, which is 6 on the ten global layers and **8 on the thirty windowed
ones** (`num_attention_heads_per_layer` = 48/64) â€” so it is wrong for 75% of the stack. `G` is
the arithmetic intensity of the attention read (`AI = 2G/b`, which for bf16 is just `G`), it
is the parameter count of `q_proj`, `o_proj` and `g_proj`, and it is the `n_q` in the
`BÂ·n_qÂ·TÂ²` score matrix that decides the prefill hang threshold. A cost model keyed on the
top-level number is right about capacity and wrong about speed. Note this is the *correction*
recorded in `ASSUMPTIONS.md â†’ laguna-heads-uniform`, whose first version overclaimed in the
other direction.

**2.** It fails before it finishes loading, and then it fails again. **First and earliest:
host RAM.** With 96 GB carved out for the GPU, Windows sees 31.6 GB; the checkpoint is
62.29 GiB. Any load path that materialises the state dict on the host â€” rather than streaming
4.77 GiB shards straight to device â€” runs out of system memory during load, and on this
platform that presents as paging and then a hang rather than a clean OOM. **Second:
activations.** Even granting that 62.29 GiB of weights lands in a tier characterised only to
62 GiB, there is nothing left. `[M]` SDPA on this stack materialises the `BÂ·n_qÂ·TÂ²` score
matrix, which is 2 GiB per windowed layer at a mere `T = 4,096`. There is no configuration in
which the bf16 artifact both fits and does useful work. The fast-tier floor argument is
correct and irrelevant â€” the binding constraint is that the *sum* of weights, KV and
transient activations must fit, and the first term already consumes the whole budget.

**3.** (a) **HuggingFace, `layer_types` absent.** `cache_utils.py:1645`'s fallback ladder
runs; if `sliding_window` is also absent every layer is full-attention and you get 163,840.
(b) **llama.cpp, `sliding_window` GGUF key absent.** `laguna.cpp:41` skips `set_swa_pattern`
entirely and builds an all-full-attention model. Same number, different program, opposite
default from (a) â€” note that with `sliding_window` present but `layer_types` absent,
`cache_utils.py:1653` makes *every* layer sliding, which is the third distinct outcome from
the same missing key. (c) **vLLM with the hybrid allocator disabled.** `FullAttentionSpec`'s
docstring (`kv_cache_interface.py:227`) says blocks are allocated for all tokens while
attention is still *computed* as sliding-window. This is the nastiest of the three because
the model's outputs are **identical** to the correct configuration â€” only the residency is
4Ã— â€” so no accuracy check can catch it.
**The single distinguishing check:** print the per-layer cache lengths, not the total.
Under (a) and (b) every layer's cache is `T` and the model's *outputs* change, because thirty
layers are now attending over the full context they were never trained on. Under (c) thirty
layers hold `T` entries but the mask still restricts attention to the last 512, so outputs
match the reference exactly. Outputs distinguish (a)/(b) from (c); the config distinguishes
(a) from (b).

**4.** No, and the "no" is exact: with `add_generation_prompt=True` the two renders differ by
**zero characters and one token id** â€” `<think>` (18) versus `</think>` (19), both single
added tokens, at the very end of the prompt. It changes more than one token in exactly one
circumstance: **preserved thinking**, where prior assistant turns carry `reasoning_content`
that the template re-renders. `chat_template.jinja:54-58` then replaces `<think>â€¦</think>`
with a bare `</think>` in *every* historical assistant message, deleting all of the retained
reasoning text from the prompt at once. For a prefix cache that is the difference between
invalidating the final partial block and invalidating everything after the first assistant
turn â€” potentially the entire conversation. The flag looks like a per-request sampling knob
and is, in the recommended usage, a cache-coherence event.

**5.** Batching multiplies the **weight** term's arithmetic intensity by `B` â€” `2B/b_w` â€” and
does not move the **attention** term at all, because a batch of `B` sequences has `B` separate
KV caches, so bytes and FLOPs both scale by `B` and the ratio is unchanged. That much is
standard. The MoE-specific inversion is on the *locality* of the weight read: at batch 1 each
layer touches 8 of 256 experts, which is a sparse gather with a small footprint; at batch 64
the union of the batch's top-k selections covers a large fraction of all 256, so the layer's
weight read approaches **dense** â€” more total bytes per step, but streamed rather than
gathered, and amortised over 64 tokens. So batching improves the term you want it to improve
(bytes per token falls sharply) by making the access pattern *less* selective, which is the
opposite of the hash-join move where you improve throughput by making probes hit fewer
partitions. The mechanism that helps here is amortisation, not locality. The cost is that
batch size and KV residency compete for the same fast tier, and on this machine the KV term
is the one with room.

**6.** Observable difference: `g_proj`'s output width. Upstream
(`.../transformers/models/laguna/modeling_laguna.py:370`) treats `True` as **per-head** â†’
width `n_q`, i.e. 64 on a windowed layer; the shipped remote code
(`models/laguna-xs/modeling_laguna.py:394`) treats `True` as per-element â†’ width
`n_q Â· d_h` = 8192. A 128Ã— difference in that projection's parameter count, and therefore a
checkpoint that will not load into the other implementation. Laguna-XS 2.1 does not exhibit
it because `config.json:40` sets `"gating": "per-head"` explicitly, so both implementations
agree â€” the default is never consulted. The one-line fix for Proteus: make the field
`Literal["off", "per-head", "per-element"]` with no boolean member and no default. A boolean
whose meaning is contested is a boolean that should not exist; the general rule is that
**any config field with more than two semantically distinct behaviours must be an enum**, and
`gating` has three.

---

## Sources

**Artifact reads (`[M]`), all 2026-07-26, all at the revisions in
`research/reference/PROVENANCE.md`.** These are static reads of pinned files, deterministic
and reproducible by anyone with the same revision â€” not runtime measurements. Revisions:
`laguna-xs` @ `205dc65dd4bd`, `laguna-s` @ `b0a9fd7c850e`, `transformers` @ `b6d5084fb4a5`,
`llama-cpp-laguna` @ `04b2b72cb540`, `vllm` @ `0934b267906f`.

- `models/laguna-xs/config.json` â€” 40 layers; `layer_types` 10 `full_attention` /
  30 `sliding_attention` in a strict GSSS period-4 pattern; `sliding_window` 512;
  `num_key_value_heads` 8; `head_dim` 128; `hidden_size` 2048; `intermediate_size` 8192;
  `num_attention_heads_per_layer` 48/64; `num_experts` 256; `num_experts_per_tok` 8;
  `moe_intermediate_size` and `shared_expert_intermediate_size` both 512; `mlp_only_layers`
  `[0]`; `vocab_size` 100352; `max_position_embeddings` 262144; `tie_word_embeddings` false;
  `torch_dtype` bfloat16; `gating` `"per-head"`; two rope configs keyed by layer type.
- `models/laguna-xs/model.safetensors.index.json` â€” `total_size` 66,885,234,176 B; 14 shards;
  expert tensors stored **unfused** as `layers.<i>.mlp.experts.<e>.{gate,up,down}_proj.weight`.
- `models/laguna-xs/model-00001-of-00014.safetensors` â€” a three-line git-lfs pointer,
  `size 5120041576`. The weights are **not** on disk.
- `models/laguna-xs/tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` â€” as cited
  inline in Â§3.6 and Â§5.5.
- `models/laguna-xs/README.md` â€” 33B total / 3B active; 40 layers, 10 global + 30 SWA;
  256 experts + 1 shared; 262,144 context; FP8 KV cache; quantised variants and GGUF repo;
  the vLLM tool-parser caveat at `:141`; the macOS Metal MoE-down-projection overflow at
  `:300`.
- `models/laguna-xs/generation_config.json` â€” `speculative_config` with
  `num_speculative_tokens: 15`, method `dflash`.
- `models/laguna-xs/{configuration,modeling}_laguna.py` and
  `architecture/transformers/src/transformers/models/laguna/{configuration,modeling}_laguna.py`
  â€” the `gating=True` divergence (Â§2.5a), the `swa_rope_parameters` derivation (Â§2.5b), the
  per-layer dispatch sites.

**Derived arithmetic (`[A]`), stated as such.** Parameter count 33,442,617,088 (Â§3.2) and
active count 2,811,614,976 (Â§3.3) are exact integer arithmetic over the `[M]` config, and the
first is *verified* against the index's `total_size` to the byte, which makes it as good as a
read. Everything with a time or a rate attached â€” the decode floor (~28â€“36 tok/s), the
prefill hang threshold (T = 16,384), the Q4_K_M size (~17.5 GiB), the FP8 size (~31.15 GiB),
transfer times, and the 2Ã— disk budget â€” is arithmetic over other people's `[M]` and has
**not been run**.

**Repo `[M]` inputs used but not re-measured here** (`ASSUMPTIONS.md`, all 2026-07-26 unless
noted): `gpu-fast-tier-size` (â‰¥62 GiB at ~199.9 GB/s with the 96 GB BIOS carve-out; single
run per arm, an anecdote by house standard); `large-tensor-fault-32gib` (â‰¥32 GiB single
tensors hang at 0% CPU or fault); `sdpa-is-memory-efficient` (147.2 vs 6.6 bytes/TÂ² retained;
`flash_sdp_enabled()` returns True either way); `hipblaslt-config` (a numerics control worth
~2.8Ã— in long-reduction accuracy, not a throughput cliff â€” **every run must record whether it
was set**); `bf16-reduced-precision-knob-works` (**refuted** â€” the knob is inert; do not use
it as an experimental axis); `gemm-throughput-below-reference` (20.9 TFLOPS bf16 at 8192Â³,
63% of the cited figure, unexplained); `single-device-only` (collectives incomplete);
`bf16-numerics-unproven` (**untested** â€” the Hardware Validation Gate has not run, so nothing
from this machine is evidence yet); `torch-build`; `reference-model`; `laguna-heads-uniform`;
`kv-per-token-laguna`; `z13-is-right-instrument` (**untested, status downgraded**).
`ENVIRONMENT.md` (31.6 GB host RAM after the carve-out; `huggingface-cli` absent).
`notebook/uma-carveout-controls-fast-tier.md` (the fast tier is a floor, not an edge).
`research/memory/kv-cache-mechanics.md` and `curriculum/kv-cache-mechanics.md` (the ~150 GB/s
decode-shaped attention read; the fp8 dequantise costing 2.9â€“3.1Ã—; `torch._scaled_mm`
unsupported on gfx1151). `research/notes/inference-and-quantization.md` (4.5 bits/weight for
`block_q4_K`; the ~105 FLOP/byte ridge; speculation as a KV-capacity consumer).

**Code pointers.** Every `file:line` in Â§5 was opened and the named construct confirmed on
the named line on 2026-07-26. Pointers reused from `research/reference/CODE_MAP.md`
(machine-verified by `scripts/generate_code_map.py`):
`architecture/llama-cpp-laguna/src/models/laguna.cpp:41`, `:154`, `:184`;
`architecture/llama-cpp-laguna/src/llama-kv-cache-iswa.cpp:73`;
`architecture/llama-cpp-laguna/src/llama-graph.cpp:2891`;
`architecture/llama-cpp-laguna/src/llama-kv-cache.cpp:319`;
`architecture/transformers/src/transformers/models/laguna/modeling_laguna.py:370`, `:185`.
Pointers reused from `curriculum/kv-cache-mechanics.md`:
`architecture/transformers/src/transformers/cache_utils.py:240`, `:1645`, `:1653`;
`memory/vllm/vllm/v1/kv_cache_interface.py:204`, `:227`;
`memory/vllm/vllm/v1/core/block_pool.py:647`.
Pointers introduced by this module and verified by reading:
`models/laguna-xs/config.json:14`, `:17`, `:27`, `:32`, `:40`, `:41`, `:42`, `:59`, `:102`, `:187`;
`models/laguna-xs/configuration_laguna.py:37`, `:189`, `:192`;
`models/laguna-xs/modeling_laguna.py:375`, `:394`, `:400`, `:427`, `:431`, `:434`, `:459`,
`:479`, `:485`, `:607`, `:616`, `:666`, `:668`, `:686`;
`models/laguna-xs/model.safetensors.index.json:3`;
`models/laguna-xs/model-00001-of-00014.safetensors:1`;
`models/laguna-xs/generation_config.json:13`, `:17`;
`models/laguna-xs/tokenizer.json:638`, `:644`, `:652`, `:665`, `:732`;
`models/laguna-xs/tokenizer_config.json:515`, `:521`, `:564`, `:567`, `:570`;
`models/laguna-xs/chat_template.jinja:3`, `:8`, `:9`, `:16`, `:44`, `:54`, `:55`, `:57`,
`:76`, `:86`;
`models/laguna-xs/README.md:4`, `:29`, `:33`, `:35`, `:43`, `:45`, `:141`, `:158`, `:217`,
`:276`, `:300`;
`architecture/transformers/src/transformers/models/laguna/configuration_laguna.py:33`.

**arXiv (`[C]`).** All four ids below appear in this repo's existing verification files
(`research/notes/citation-verification.json`, `curriculum/citation-verification.json`),
resolved against the live arXiv API on 2026-07-26. Resolving an id proves the paper exists,
not that it supports the claim beside it.

- `2309.06180` â€” *Efficient Memory Management for LLM Serving with PagedAttention*
  (2023-09-12). The block allocator behind Â§5.4.
- `2309.17453` â€” *Efficient Streaming Language Models with Attention Sinks* (2023-09-29).
  The mechanism `swa_attention_sink_enabled` implements and this checkpoint does not use.
- `2602.06036` â€” *DFlash: Block Diffusion for Flash Speculative Decoding* (2026-02-05).
  ICML 2026. The speculator named in `generation_config.json`. Author-reported speedups; not
  independently replicated.
- `2605.05699` â€” *When Quantization Is Free: An int4 KV Cache That Outruns fp16 on Apple
  Silicon* (2026-05-07). The unified-memory counter-result in unsolved item 5.

**Non-arXiv.** The Laguna XS 2.1 model card and the OpenMDW-1.1 license, both in the clone.
The Poolside technical report linked at `README.md:29` (not fetched). vLLM PRs #41129 and
#47311, SGLang #24204, llama.cpp #25165, TensorRT-LLM #13559 â€” all cited by the model card,
none verified here.

**Mirrored notes.** This module does not mirror a single survey; it draws on
`research/notes/inference-and-quantization.md` v1.0.0 and `research/memory/kv-cache-mechanics.md`
v1.0.0. No number here contradicts either. It supplies one input those notes lack: the
XS-scale instantiation of the KV geometry (40,960 B/token growing, 62,914,560 B fixed) and the
exact parameter decomposition that closes against `total_size`. It adds one open question to
`research/notes/inference-and-quantization.md`'s list â€” the MoE gather bandwidth in unsolved
item 3 â€” which is measurable today, without weights, and is the cheapest remaining test of
`ASSUMPTIONS.md â†’ z13-is-right-instrument`.
