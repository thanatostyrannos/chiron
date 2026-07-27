---
title: Supervised and preference finetuning — SFT, DPO and its successors, reward models vs verifiers, GRPO and RLVR
slug: supervised-and-preference-finetuning
version: 1.0.0
date: 2026-07-26
track: E — Post-training and evaluation
owner: curriculum-author
prereqs: the-training-loop, loss-and-optimization, tokenization, kv-cache-mechanics
recommended: paged-attention-and-prefix-reuse, scaling-laws-and-flops-budget
difficulty: medium for the algebra, hard for the discipline — the whole module argues against running most of what it teaches
time: 3–4 h reading and working the arithmetic; 2–4 h for the exercises
mirrors: research/notes/posttraining-pipelines.md
---

# Supervised and preference finetuning

**Scope boundary, stated first.** This module teaches post-training *mechanisms*: what
each loss is, what signal it consumes, what it costs, and how it fails. It does **not**
teach how to build an evaluation you can trust — that is the other Track E module, and the
survey behind it is `research/notes/evaluation-landscape.md`. It does not re-teach the
training loop, cross-entropy, the logits allocation, or the KV-cost arithmetic; those are
`the-training-loop`, `loss-and-optimization`, and `kv-cache-mechanics`, and this module
leans on all three without repeating them.

**Difficulty and time, honestly.** The algebra in section 3 is one page of calculus you
already have: a logistic loss, a chain rule, and a binomial. Budget 3–4 hours to work it
with a pen. What makes the module hard is that it is mostly a *refusal*: the honest
conclusion for a 20M–300M single-GPU lab is that SFT and preference optimization are
fully ours, and reinforcement learning from verifiable rewards is not — not because of
memory, which we have in abundance, but because the reward signal collapses at our
capability level, and the collapse is arithmetic you can do before writing any code. Read
this as paper literacy, not as a recipe.

---

## 1. What this module settles

**One:** post-training is four different objectives wearing one word, and they are
distinguished by *what produces the training signal* — a human-written demonstration
(SFT), a human-written comparison (reward model / DPO), or a program that returns a bit
(RLVR) — with cost, coverage, and failure mode following mechanically from that one
choice. **Two:** the entire DPO family is a single algebraic move — substitute the policy
for the reward model inside the Bradley–Terry likelihood so the partition function cancels
— and every successor is named for the term it deletes; but the same algebra shows the
loss constrains only the *margin* between two log-ratios and nothing anchors their absolute
level, which is why chosen-response likelihood routinely falls during DPO and why length
bias is not a data artifact but a property of summing per-token log-probabilities.
**Three:** GRPO at our scale is blocked by advantage collapse, and the block is
quantitative rather than vague — with group size `G = 8` and a base-model pass rate of
1.82%, **86.3% of prompt groups produce exactly zero gradient**, which is the mechanism
behind the published single-GPU 135M result where GRPO-style RLVR *decreased* GSM8K exact
match from 1.82% to 1.59% `[C]`
([2606.22189](https://arxiv.org/abs/2606.22189), Jun 2026).

---

## 2. Theory in plain language

### 2.1 What post-training is, and what it replaced

A pretrained base model is a conditional distribution over next tokens fitted to a corpus.
It is not a system that answers questions; it is a system that continues text. If the
corpus contains a lot of question-answer text, prompting it in that shape gets you
something answer-like, and for a while that was the whole technique — few-shot prompting
`[C]` ([2005.14165](https://arxiv.org/abs/2005.14165)). Post-training replaced prompting
with weight updates that make the answering behaviour the default rather than a trick.

Four stages, in the order the field discovered them, each defined by its signal source:

| Stage | Signal | What it is mechanically |
|---|---|---|
| **SFT** | a human (or a stronger model) wrote the desired output | cross-entropy, masked to the response span |
| **Reward modelling + RL** | a human said "A is better than B" | fit a scalar scorer, then optimize the policy against it |
| **Preference optimization (DPO family)** | same pairs, no RL | one logistic loss directly on the policy |
| **RLVR** | a *program* said the answer is right | policy gradient with a 0/1 reward from a verifier |

The `research/notes/posttraining-pipelines.md` survey adds a fifth — agentic RL, where the
episode is a multi-turn interaction with a live environment. That one is out of reach here
and section 8 says so plainly rather than shipping a toy.

**Systems bridge that half-works: post-training is a build-and-deploy chain.** Each stage
takes the previous artifact, applies a transform, emits a new one. That framing pays in
one specific way — each stage has a distinct *input contract*, and the field's entire cost
structure follows from how expensive that input is to produce. Demonstrations are the most
expensive per unit; comparisons are cheaper because judging is easier than writing;
verifier bits are nearly free and, unlike human labels, **do not decay** as the policy
changes.

**Where it breaks, and this is not a small break.** Build stages are idempotent and
composable. Post-training stages *regress each other*. RL for reasoning routinely degrades
instruction-following and safety behaviour, and the remedy is not a rollback but a
rebalanced data mix in the next full run — Tulu 3 documents that loop explicitly `[C]`
([2411.15124](https://arxiv.org/abs/2411.15124)), and Olmo 3 publishes the most complete
SFT → DPO → RLVR flow available `[C]`
([2512.13961](https://arxiv.org/abs/2512.13961), Dec 2025). There is no `git revert` here:
the artifact is a weight tensor, the regression is distributed across all of it, and the
only "rollback" is retraining from an earlier checkpoint with different data. Treat a
post-training pipeline as a **non-idempotent, non-commutative sequence of destructive
transforms whose only recovery mechanism is a full rebuild**, and you will predict its
operational behaviour correctly.

A second folklore item worth killing early: "SFT is the cheap safe stage, RL is the risky
one." Contested. "RL's Razor" argues online RL *forgets less* than SFT at matched task
gain, because an on-policy update stays inside the model's own distribution while SFT drags
it toward an external one `[C]` ([2509.04259](https://arxiv.org/abs/2509.04259)).

### 2.2 SFT is cross-entropy plus an access-control list

Mechanically SFT is the loss you already know from `loss-and-optimization`, computed over a
conversation, with the gradient **masked** so it flows only through the tokens the
assistant is supposed to produce. Everything else — system prompt, user turns, tool
outputs — is context that conditions the prediction but never receives a gradient.

That masking decision is the whole of SFT engineering, and it is consistently
under-documented. The operationally important fact: **the mask is defined by the chat
template, which ships inside the model artifact, not by your trainer.** In HuggingFace's
implementation the template author marks the assistant span with Jinja `{% generation %}`
… `{% endgeneration %}` tags, a custom Jinja extension records the character offsets of
those spans during rendering, and the tokenizer converts offsets to token indices
afterwards. Section 6 walks the four files.

**Systems bridge: the mask is a column-level ACL on a write path.** The template is the
policy file; the trainer is the enforcement point; the gradient is the write.

**Where it breaks, three ways, all of which have cost someone a run.**

1. **An ACL denies loudly. A mask fails silently.** If the template has no
   `{% generation %}` markers, HuggingFace emits a `warning_once` and returns an *empty*
   list of spans — so the mask is all zeros, the loss over the assistant span is empty, and
   depending on your reduction you get either `nan` or a loss of zero that trains nothing.
   A warning, not an exception.
2. **The policy file is versioned outside your repo, and there can be more than one copy
   that disagree.** In this repository there are two shipped copies of the Laguna S 2.1
   chat template. The one in the HF model repo defaults `enable_thinking` to **true** and
   carries an extra `preserve_thinking` flag; the one vendored into llama.cpp defaults it
   to **false** and has no such flag. Section 6 gives both `file:line`. The rendered
   training sequence — and therefore the mask, and therefore what the model learns — is
   different depending on which copy your loader picked up.
3. **The mask changes the denominator of your token budget and nobody adjusts for it.**
   The `6ND` FLOPs rule from `scaling-laws-and-flops-budget` counts every token you push
   through the model. The mask means only a fraction of those receive gradient. Section 3.1
   does the arithmetic; the correction is typically 2–3×, and it is the difference between
   "we ran a 1B-token SFT" and "we ran a 350M-gradient-token SFT."

### 2.3 Reward models: a learned cache in front of an expensive oracle

The original RLHF loop is three models and two losses `[C]`
([2009.01325](https://arxiv.org/abs/2009.01325),
[2203.02155](https://arxiv.org/abs/2203.02155)). First, collect pairs of responses to the
same prompt and have a human say which is better. Fit a scalar function `r_φ(x, y)` — a
copy of the base model with its `V`-wide output head replaced by a **one**-wide one — under
a Bradley–Terry likelihood. Then optimize the policy to maximize `r_φ` with PPO `[C]`
([1707.06347](https://arxiv.org/abs/1707.06347)) plus a KL penalty pulling it back toward
a frozen reference.

**Systems bridge, from `research/notes/posttraining-pipelines.md` §3 and worth restating
because it is the sharpest one in this module: the reward model is a learned cache in
front of an expensive oracle (the human).** You cannot call the oracle at 10 kHz inside a
training loop, so you fit an approximation and call that instead.

**Where it breaks, and the break is the entire failure mode.** A cache validates against
the origin on a miss. This one has **no miss signal**. Worse, the access pattern is
adversarial and *you generated it yourself*: the policy is explicitly optimizing to find
inputs where the cache returns a high value, which is precisely the region where the cache
is least likely to agree with the origin. That is reward overoptimization, it follows a
smooth and fittable curve, and the only detector is a separate evaluation run later on a
different distribution `[C]` ([2210.10760](https://arxiv.org/abs/2210.10760)).

### 2.4 DPO: deleting the online service from the training loop

DPO's insight is algebraic rather than empirical. Under the KL-regularized RL objective the
optimal policy has a closed form in terms of the reward; invert it and the reward has a
closed form in terms of the optimal policy. Substitute that expression into the
Bradley–Terry likelihood and the intractable per-prompt partition function `Z(x)` cancels,
because it appears identically in both terms of a comparison over the *same* prompt. What
is left is a plain logistic loss on two log-probability ratios — no reward model, no
sampling, no value network `[C]`
([2305.18290](https://arxiv.org/abs/2305.18290)). Section 3.3 does the derivation.

**Systems bridge: DPO deletes the online serving path from the training loop and replaces
it with a batch job over a static table.** The reward model was a service you called; now
it is a materialized view — the reference log-probabilities, which are a pure function of
(frozen reference model, dataset) and can be computed once and cached to disk. Section 3.6
prices that view: for 100,000 preference pairs it is **781 KiB**, and it removes a 600 MB
resident model and half the forward passes per step.

**Where it breaks.** A materialized view has a validity key, and here the key is the
reference policy. Vanilla DPO is offline: the pairs were generated by some other policy,
scored once, and never regenerated. The consequence is not a stale read — it is that the
optimizer can only reweight mass that the reference policy already places somewhere. It
cannot discover a response neither `y_w` nor `y_l` contains, and pushing down `y_l` moves
mass to wherever the model already had it, which is frequently nowhere useful. This is the
formal reason iterative and online DPO variants exist `[C]`
([2506.21495](https://arxiv.org/abs/2506.21495),
[2405.19320](https://arxiv.org/abs/2405.19320),
[2404.19733](https://arxiv.org/abs/2404.19733)) — and every one of them invalidates the
materialized view on every round, which is the systems cost of recovering the on-policy
property.

**Contested, and left contested:** whether offline preference optimization is a real stage
in 2026 or a cheap legacy step. Olmo 3 keeps DPO between SFT and RLVR `[C]`
([2512.13961](https://arxiv.org/abs/2512.13961)); R1-style reasoning recipes go straight
from an SFT cold start to RL `[C]`
([2501.12948](https://arxiv.org/abs/2501.12948)). **No matched-budget ablation isolating
what the DPO stage contributes when RLVR follows it has been published.** That is a genuine
hole and — see section 8 — it is a small-scale-shaped hole.

### 2.5 RLVR: replacing the learned reward with a program

RLVR's move is to stop learning the reward at all. Pick tasks where correctness is
*checkable*: the answer matches the reference string, the unit tests pass, the compiler
accepts it, the proof checker verifies it. Reward is 1 or 0. Tulu 3 named and systematized
this `[C]` ([2411.15124](https://arxiv.org/abs/2411.15124)); DeepSeek-R1 and Kimi k1.5
demonstrated it at frontier scale `[C]`
([2501.12948](https://arxiv.org/abs/2501.12948),
[2501.12599](https://arxiv.org/abs/2501.12599)).

| | Learned reward model | Verifiable reward |
|---|---|---|
| Signal source | human/AI preference pairs | a program |
| Failure mode | Goodhart drift, silent, unreproducible | verifier gaming — also silent, but **reproducible** |
| Cost per label | high, and it decays as the policy moves | near zero, and it does not decay |
| Coverage | any task | only tasks with a checkable answer |
| Density | one scalar per response | **one bit at the end of a 10,000-token trajectory** |

**Systems bridge: a verifiable reward is a checksum.** Cheap, deterministic, catches the
error class it was designed for.

**Where it breaks, and the break is not subtle.** A checksum has no *incentive* to be
satisfied without the underlying property holding. A policy does. Extensional checking
admits false positives, and gradient descent is an extremely effective search for them. A
2026 study shows RLVR-trained models systematically abandoning rule *induction* in favour
of enumerating instance labels that happen to pass the checker, with
isomorphic-perturbation verification removing the shortcut `[C]`
([2604.15149](https://arxiv.org/abs/2604.15149), Apr 2026). Verifiers themselves are now
fuzzed `[C]` ([2606.01066](https://arxiv.org/abs/2606.01066), May 2026), and noisy
verifiers are a first-class research object `[C]`
([2510.00915](https://arxiv.org/abs/2510.00915),
[2603.16140](https://arxiv.org/abs/2603.16140)). Read the verifier code in section 6 before
you believe any RLVR number: `exact_match` in lm-evaluation-harness is literally
`predictions == references`, and the ~450 lines of LaTeX normalization that make that `==`
mean something include a fallback of "if nothing else parses, take the last number in the
output."

**GRPO** is the algorithm that made RLVR cheap `[C]`
([2402.03300](https://arxiv.org/abs/2402.03300)). For each prompt sample `G` completions,
score them all, and use the group's own mean as the baseline. The value network — half of
PPO's memory — disappears, and so does the reward model. Section 3.7 writes the objective
out; section 3.8 does the collapse arithmetic that decides whether we can run it.

**Systems bridge for GRPO's group baseline: it is a canary deployment with the control
group in the same batch.** You do not need a separately-maintained baseline service; the
siblings are the control.

**Where it breaks, and this break *is* our blocker.** When a canary and its control behave
identically, an A/B test reports "no measurable difference." GRPO reports **nothing at
all** — a group whose `G` rewards are identical has zero advantage on every token and
contributes exactly zero gradient. It is not a null result; it is a silently discarded
sample. The batch you configured is not the batch you trained on, no error is raised, and
the only symptom is that learning is slower than it should be. Section 3.8 turns that into
a number.

---

## 3. The math that actually matters

### 3.1 The SFT loss and what the mask does to your token budget

For one training sequence of `T` tokens with labels `y_1..y_T` and a binary mask
`m_1..m_T`:

```
L_SFT = ( Σ_{t=1..T} m_t · [ logsumexp(z_t) − z_{t,y_t} ] ) / ( Σ_{t=1..T} m_t )
```

| Symbol | In words |
|---|---|
| `T` | tokens in the rendered conversation, including system prompt and user turns |
| `z_t` | the logit vector at position `t`, one entry per vocabulary item |
| `y_t` | the token that actually appears at position `t+1` (the next-token label) |
| `m_t` | **1 if position `t` is inside an assistant span, 0 otherwise** — this is the whole of SFT |
| `logsumexp(z_t) − z_{t,y_t}` | ordinary next-token cross-entropy in nats (`loss-and-optimization` §*Cross-entropy*) |

Nothing here is new except `m`. The numerator sums loss only over assistant tokens; the
denominator counts only assistant tokens, so the reported loss is per *gradient* token.

**The budget correction.** Define the **mask fraction** `μ = (Σ m_t) / T`. Then:

```
gradient tokens delivered = μ · (tokens processed)
FLOPs per gradient token   = 6·N / μ
```

Worked, at our ablation scale (`N = 300M`) on a dataset packed to 4096-token sequences:

| Data shape | typical `μ` | FLOPs per gradient token |
|---|---|---|
| short instruction pairs, long answers | ~0.60 | `6N / 0.60` = 10·N |
| conversational, several turns | ~0.35 | `6N / 0.35` = 17.1·N |
| agentic trajectory, long tool outputs | ~0.15 | `6N / 0.15` = 40·N |

At `μ = 0.35` a nominal 1B-token SFT run delivers **350M gradient tokens** and costs
`6 × 3e8 × 1e9 = 1.8e18` FLOPs. Against our measured `[M]` 20.9 TFLOP/s bf16 GEMM peak
(`ASSUMPTIONS.md: gemm-throughput-below-reference`, 8192³, 2026-07-26) that is
`1.8e18 / 2.09e13 = 86,124 s ≈ 23.9 h` **at 100% of measured GEMM peak**, which nothing
achieves. Assume `[A]` 20–35% of that peak in practice — low confidence, since the
Hardware Validation Gate has not run a known-good recipe on this machine — and a 1B-token
SFT at 300M is a `[A]` 3–5 day job. A realistic SFT dataset is far smaller: 100,000
examples at 512 tokens is 51.2M tokens, `9.2e16` FLOPs, `[A]` **4–8 hours**. That is the
honest SFT budget on the Z13, and it is affordable.

**Note the asymmetry the mask creates in the memory budget: it creates none.** The logits
tensor is `T × V` whether or not the tokens are masked — you compute the full output
projection and then throw away the masked rows' contributions. The mask saves gradient
signal accounting, not memory. If you want the memory back you must slice *before* the
`lm_head`, which almost no trainer does.

### 3.2 Bradley–Terry and the reward model

Given a prompt `x` and two responses, the Bradley–Terry model says the probability a
labeller prefers `y_w` over `y_l` is a logistic function of the reward difference:

```
P(y_w ≻ y_l | x) = exp(r(x,y_w)) / ( exp(r(x,y_w)) + exp(r(x,y_l)) ) = σ( r(x,y_w) − r(x,y_l) )
```

| Symbol | In words |
|---|---|
| `y_w` | the "chosen"/winning response |
| `y_l` | the "rejected"/losing response |
| `r(x,y)` | a scalar score for this response to this prompt |
| `σ(u)` | the logistic function `1/(1+e^{−u})` |

The training loss is the negative log-likelihood of the observed preferences:

```
L_RM(φ) = − E_{(x, y_w, y_l) ~ D} [ log σ( r_φ(x,y_w) − r_φ(x,y_l) ) ]
```

**The structural fact that matters, and it is the same one as in
`loss-and-optimization` §*Shift invariance*.** Only *differences* of `r` appear. Add any
per-prompt constant `c(x)` to every reward for that prompt and the loss is unchanged.
Therefore:

- **A reward model's absolute output is meaningless.** "The reward went from 0.8 to 2.3"
  says nothing without stating what it is being compared against on the same prompt.
- **Reward scores are not comparable across prompts.** A run that reports "mean reward" as
  a training curve is averaging quantities with independent, unconstrained offsets. It can
  go up because the prompt mix shifted.
- The same fix applies as for logits: constrain the free degree of freedom explicitly, or
  only ever report within-prompt differences.

**Architecture.** The reward model is the base model with the `d × V` output matrix
replaced by a `d × 1` one, and the score read at the final non-padding token. For Laguna's
`d = 3072` and `V = 100352`, that is 3,072 new parameters replacing 308M — **the reward
model is 0.001% new weights and 99.999% the thing it is scoring.** Section 6 points at the
four lines that implement this, including the pooling index that silently reads the wrong
position when `pad_token_id == eos_token_id`.

### 3.3 DPO, derived

Start from the KL-regularized RLHF objective:

```
max_π  E_{x~D, y~π(·|x)} [ r(x,y) ]  −  β · KL( π(·|x) ‖ π_ref(·|x) )
```

| Symbol | In words |
|---|---|
| `π(y|x)` | the policy being trained — probability it assigns to the whole response `y` |
| `π_ref(y|x)` | a frozen reference, normally the SFT checkpoint you started from |
| `β` | how hard the KL term pulls the policy back toward the reference |
| `KL(π‖π_ref)` | expected log-ratio `E_{y~π}[log(π(y|x)/π_ref(y|x))]` — "how far the policy has moved" |

This has a known closed-form maximizer:

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r(x,y) / β )        with  Z(x) = Σ_y π_ref(y|x) exp(r(x,y)/β)
```

`Z(x)` is a sum over every possible response — astronomically intractable, which is why
nobody uses this form directly. But solve for `r`:

```
r(x,y) = β · log( π*(y|x) / π_ref(y|x) )  +  β · log Z(x)
```

Now substitute into Bradley–Terry. The two responses share the prompt `x`, so they share
`Z(x)`, and it **cancels in the difference**:

```
P(y_w ≻ y_l | x) = σ(  β·log(π(y_w|x)/π_ref(y_w|x))  −  β·log(π(y_l|x)/π_ref(y_l|x))  )
```

which gives the DPO loss:

```
L_DPO(θ) = − E [ log σ( β·Δ_w − β·Δ_l ) ]

    where  Δ_w = log π_θ(y_w|x) − log π_ref(y_w|x)
           Δ_l = log π_θ(y_l|x) − log π_ref(y_l|x)
```

| Symbol | In words |
|---|---|
| `Δ_w` | how much more likely the policy makes the chosen response than the reference did |
| `Δ_l` | the same for the rejected response |
| `β·Δ` | the **implicit reward** `r̂` — DPO's reward model is the policy itself |
| `β` | typically 0.1; larger keeps you nearer the reference |

`log π(y|x)` is the **sum** of per-token log-probabilities over the response — which is
exactly `−1 ×` the sum of the per-token cross-entropies you already compute. This is the
concrete reason `loss.reduction="none"` is a hard interface requirement and not a
convenience (section 5).

### 3.4 The DPO gradient, and why chosen likelihood falls

Let `u = β(Δ_w − Δ_l)` be the margin. Since `d/du log σ(u) = σ(−u)`:

```
∇_θ L_DPO = − β · E [ σ( −u ) · ( ∇_θ log π_θ(y_w|x)  −  ∇_θ log π_θ(y_l|x) ) ]
```

| Term | In words |
|---|---|
| `σ(−u)` | the per-example weight: **how wrong the implicit reward model currently is.** Near 1 when the model ranks the pair backwards, near 0 once the margin is large |
| `∇log π(y_w)` | the direction that makes the chosen response more likely |
| `−∇log π(y_l)` | the direction that makes the rejected response less likely |

Three consequences, all mechanical.

**(a) Only the margin is optimized. Nothing anchors the level.** The loss is a function of
`Δ_w − Δ_l` alone. `Δ_w = −3, Δ_l = −8` scores identically to `Δ_w = +2, Δ_l = −3`. So
"chosen log-probability went down during DPO" is not a bug report — it is entirely
consistent with the loss decreasing, and it is what usually happens. The reason it usually
happens rather than sometimes: `y_w` and `y_l` are two plausible answers to the same
prompt, so they share most of their tokens and most of their probability mass. Pushing
down on `y_l` pushes down on shared structure, and the chosen response is made of that same
structure. The margin widens while both terms sink.

**(b) The displaced mass has to go somewhere, and it is not the dataset.** Probabilities
sum to one. Mass removed from `y_l` (and incidentally from `y_w`) is redistributed over
sequences that appear in neither. Nothing in the loss constrains where. `β` and the
reference term are the only brakes, and they are weak: the KL is implicit, not enforced.
This is the mechanism behind the off-policy support limit `[C]`
([2506.21495](https://arxiv.org/abs/2506.21495)) and it is why measuring "did the chosen
log-prob go up" is a more informative diagnostic than "did the loss go down."

**(c) The weight `σ(−u)` is a self-annealing curriculum, with a failure mode.** Examples
the model already ranks correctly contribute almost nothing. That sounds good — it is
focal-loss-like — until you notice it means DPO's effective batch size shrinks as training
progresses, and the remaining gradient is dominated by pairs the model gets wrong, which at
small scale are disproportionately *mislabelled* pairs. Label noise is amplified by
construction.

### 3.5 Length bias is arithmetic, not a data artifact

`log π(y|x) = Σ_{t=1..|y|} log π(y_t | x, y_{<t})`. Every term is negative. So:

```
Δ_w = Σ_{t=1..|y_w|} [ log π_θ(y_{w,t}|·) − log π_ref(y_{w,t}|·) ]  ≈  |y_w| · δ̄_w
```

where `δ̄_w` is the *average per-token* log-ratio over the chosen response and `|y_w|` is
its token count. The implicit reward is therefore **approximately proportional to response
length**, with the per-token log-ratio as the constant of proportionality.

Concretely: suppose the policy has drifted so that its average per-token log-ratio against
the reference is `δ̄ = +0.01` nats on responses of the style it is being pushed toward.
Then a 200-token response earns implicit reward `β × 2.0` and an 800-token response earns
`β × 8.0`, from *nothing but length*. The optimizer can increase the margin by making
preferred-style outputs longer, without improving anything a human would call quality.

This is the derivation behind the empirical finding that a large fraction of measured RLHF
preference gain is explained by length alone `[C]`
([2310.03716](https://arxiv.org/abs/2310.03716)), behind ODIN's two-head reward that
disentangles it `[C]` ([2402.07319](https://arxiv.org/abs/2402.07319)), and behind SimPO's
fix — divide by the length:

```
L_SimPO = − log σ(  (β/|y_w|)·log π_θ(y_w|x)  −  (β/|y_l|)·log π_θ(y_l|x)  −  γ  )
```

`γ` is a target margin: the loss stops pushing once the gap exceeds it. Note SimPO has no
`π_ref` at all — it drops both the reference model and the length term in one move `[C]`
([2405.14734](https://arxiv.org/abs/2405.14734)).

**Practical consequence for any DPO run at our scale:** report mean chosen length and mean
rejected length per step alongside the loss. If the margin is improving and the length
ratio is drifting, you are measuring the length term, not the preference.

### 3.6 The successor zoo, and the reference-model materialized view

Each successor is named for the term it deletes. This is the whole taxonomy:

| Method | What it removes or replaces | Claim | Cite |
|---|---|---|---|
| **IPO** | the logistic loss → squared loss to a finite target | logistic loss is unbounded in the margin, so with deterministic preference labels DPO drives the margin to infinity and the KL constraint stops binding; a squared loss has a finite optimum | `[C]` [2310.12036](https://arxiv.org/abs/2310.12036) |
| **KTO** | the *pair* | preference pairs are the expensive part; unpaired desirable/undesirable labels with a prospect-theoretic value function are far cheaper to collect | `[C]` [2402.01306](https://arxiv.org/abs/2402.01306) |
| **ORPO** | the reference model *and* the separate stage | folds an odds-ratio penalty directly into the SFT loss; one pass, one model | `[C]` [2403.07691](https://arxiv.org/abs/2403.07691) |
| **SimPO** | the reference model *and* the length term | length-normalized implicit reward with a target margin | `[C]` [2405.14734](https://arxiv.org/abs/2405.14734) |
| **CPO** | the reference forward pass, keeping an SFT anchor | memory and speed at near-DPO quality | `[C]` [2401.08417](https://arxiv.org/abs/2401.08417) |
| **Iterative / online DPO** | the *offline* assumption | regenerate pairs from the current policy each round, recovering the on-policy property | `[C]` [2404.19733](https://arxiv.org/abs/2404.19733), [2402.04792](https://arxiv.org/abs/2402.04792), [2401.10020](https://arxiv.org/abs/2401.10020) |

**Now the systems arithmetic that makes DPO cheap, which no paper states because it is
implementation trivia to them and load-bearing to us.**

Vanilla DPO appears to need four forward passes per step: policy on chosen, policy on
rejected, reference on chosen, reference on rejected. But `log π_ref(y|x)` depends only on
the frozen reference and the fixed dataset. **It is a pure function of two immutable
inputs, so compute it once and store it.**

```
view rows      = 2 floats per preference pair  (Σ_t log π_ref for chosen and for rejected)
100,000 pairs  = 100,000 × 2 × 4 B = 800,000 B = 781 KiB
```

Per-step savings at `N = 300M`:

```
resident memory removed:  300e6 params × 2 B (bf16 reference)  =  600 MB
forward passes per step:  4 → 2                                 =  50% of step FLOPs
```

**A 781 KiB table removes a 600 MB model and half the compute.** That is the single best
cost/benefit ratio in this module.

**Where the analogy breaks — and it is the classic one.** A materialized view needs an
invalidation key. Here it is the tuple `(reference checkpoint hash, tokenizer hash, dataset
hash, template hash)`. Three of those four are easy to get wrong: the tokenizer and
template are shipped inside the model artifact and can change between revisions (§2.2 point
2), and iterative DPO regenerates the pairs every round, invalidating the whole view every
round. If you build this, hash all four into the filename. A silently stale reference table
produces a DPO run that trains perfectly happily against the wrong `π_ref` and reports
nothing unusual.

**The logits hazard, inherited from `loss-and-optimization`.** DPO scores two full
sequences per example, so the logits tensor is **twice** SFT's at the same example count.
Using that module's measured coefficient of ~12 bytes per logit element and `V = 100352`:

| pairs/step | tokens per response | total scored tokens | fp32 logits (single largest tensor) | transient total @12 B/elem |
|---|---|---|---|---|
| 8 | 512 | 8,192 | 3.06 GiB | 9.19 GiB |
| 8 | 2048 | 32,768 | 12.25 GiB | 36.75 GiB |
| 16 | 2048 | 65,536 | 24.50 GiB | 73.50 GiB |
| 21 | 2048 | 86,016 | **32.15 GiB — over the fault** | — |

`[M]` A single tensor of 32 GiB hard-hangs this machine at 0% CPU with no error and must be
force-killed (`ASSUMPTIONS.md: large-tensor-fault-32gib`, 2026-07-26). **At `V = 100352`
and 2048-token responses, 21 preference pairs per step is the boundary.** The row above it
already exceeds the `[M]` ≥62 GiB fast tier
(`ASSUMPTIONS.md: gpu-fast-tier-size`), so you will hit degraded bandwidth first — but the
failure at the boundary is a silent hang, not an OOM. Chunk the log-probability computation
over the sequence axis, or use the fused path (§6, `lm_head.py:268`), and assert the guard.

### 3.7 GRPO, written out

Sample `G` completions `o_1..o_G` for prompt `x` from the current policy `π_old`, score
each with the verifier to get `r_1..r_G`, and set:

```
Â_i = ( r_i − mean(r_1..r_G) ) / ( std(r_1..r_G) + ε )
```

broadcast to **every token** of completion `o_i`. The objective:

```
J(θ) = E [ (1/G) Σ_{i=1..G} (1/|o_i|) Σ_{t=1..|o_i|}
             min( ρ_{i,t}·Â_i , clip(ρ_{i,t}, 1−ε_c, 1+ε_c)·Â_i ) ]
       −  β · D̂_KL[ π_θ ‖ π_ref ]

  with  ρ_{i,t} = π_θ(o_{i,t} | x, o_{i,<t}) / π_old(o_{i,t} | x, o_{i,<t})
```

| Symbol | In words |
|---|---|
| `G` | group size — completions sampled per prompt, typically 8–64 |
| `r_i` | the verifier's score for completion `i`: 1 if it passed, 0 if not |
| `Â_i` | the advantage — how much better than its siblings this completion was, in units of the group's own standard deviation |
| `ρ_{i,t}` | the importance ratio: how much more likely the *current* weights make this token than the weights that generated it |
| `clip(·, 1−ε_c, 1+ε_c)` | trust region — refuse to take credit for a ratio that has moved too far, `ε_c ≈ 0.2` |
| `1/|o_i|` | per-response length normalization — **contested, see below** |
| `β · D̂_KL` | a penalty pulling the policy back toward the frozen reference |

The KL estimator GRPO uses is the unbiased, always-non-negative "k3" form, computed per
token from samples of `π_θ`:

```
D̂_KL = ( π_ref/π_θ ) − log( π_ref/π_θ ) − 1
```

**What GRPO deleted, in bytes.** At `N = 300M` in bf16 with fp32 AdamW moments
(16 bytes/param for anything trained, 2 bytes/param for anything frozen — from
`loss-and-optimization` §*SGD → Adam → AdamW*):

| Component | PPO | GRPO |
|---|---|---|
| policy (trained) | 4.8 GB | 4.8 GB |
| frozen reference | 0.6 GB | 0.6 GB |
| reward model (frozen) | 0.6 GB | **0** — it is a Python function |
| value network (**trained**) | 4.8 GB | **0** — the group mean is the baseline |
| **total** | **10.8 GB** | **5.4 GB** |

Exactly half the memory and two fewer model artifacts to keep in sync. That is why GRPO
won.

**Two of the most-copied lines in every GRPO implementation are contested.**

- Dr. GRPO argues both the `/std` term and the `1/|o_i|` normalization bias the gradient,
  and that removing them changes the conclusions people draw about "aha moments" `[C]`
  ([2503.20783](https://arxiv.org/abs/2503.20783)).
- DAPO independently switches to a token-level loss normalized over the whole batch, adds
  an asymmetric clip (`ε_high > ε_low`, "clip-higher") to stop low-probability tokens being
  crushed, adds dynamic sampling, adds overlong reward shaping, and **removes the KL term
  entirely** `[C]` ([2503.14476](https://arxiv.org/abs/2503.14476)).
- GSPO replaces the token-level importance ratio with a length-normalized sequence-level
  one, motivated specifically by instability when the policy is an MoE `[C]`
  ([2507.18071](https://arxiv.org/abs/2507.18071)) — directly relevant here, since our
  reference architecture is an MoE and a `proteus-moe-*` arm would inherit the problem.

**Why `1/|o_i|` inflates length, derived.** A wrong completion in a mixed group has some
negative advantage `Â < 0`. Its per-token gradient weight is `Â/|o_i|`. So a wrong response
of 800 tokens receives `Â/800` per token while a wrong response of 100 tokens receives
`Â/100` — **eight times more penalty per token for being briefly wrong than for being
verbosely wrong.** Symmetrically, a correct short answer gets more per-token reinforcement
than a correct long one. Which term dominates depends on the correct/incorrect mix, and at
low pass rates almost every sample is wrong — so the "be verbose when you expect to be
wrong" pressure dominates precisely in the regime a small model lives in. `[A]` high
confidence in the derivation; it is two lines of algebra. It is a *prediction*, not a
measurement, and it is directly testable at our scale (section 8).

### 3.8 Advantage collapse — the arithmetic that decides our scale

Assume a binary verifier and per-sample success probability `p` for this prompt under the
current policy. Successes in a group of `G` are `Binomial(G, p)`. The group's advantage is
identically zero **iff** all `G` scored the same:

```
P(zero gradient from this prompt)  =  p^G  +  (1−p)^G

fraction of prompts contributing   =  1 − p^G − (1−p)^G

dynamic-sampling rollout tax       =  1 / (1 − p^G − (1−p)^G)
```

| Symbol | In words |
|---|---|
| `p` | probability that one sampled completion passes the verifier — i.e. the model's pass@1 on this prompt |
| `G` | group size |
| `p^G` | probability all `G` succeeded (nothing to learn: everything is equally good) |
| `(1−p)^G` | probability all `G` failed (nothing to learn: everything is equally bad) |
| rollout tax | how many rollouts DAPO's dynamic sampling must generate to fill one usable batch `[C]` ([2503.14476](https://arxiv.org/abs/2503.14476)) |

At `G = 8`:

| pass rate `p` | P(degenerate) | fraction contributing | rollout tax |
|---|---|---|---|
| 0.0182 (**the published 135M GSM8K figure**) | **0.8633** | **0.1367** | **7.32×** |
| 0.05 | 0.6634 | 0.3366 | 2.97× |
| 0.10 | 0.4305 | 0.5695 | 1.76× |
| 0.25 | 0.1001 | 0.8999 | 1.11× |
| 0.50 | 0.0078 | 0.9922 | 1.01× |
| 0.75 | 0.1001 | 0.8999 | 1.11× |
| 0.90 | 0.4305 | 0.5695 | 1.76× |

**Read the first row carefully, because it is this module's central quantitative claim.**
`[C]` The L20-Edu-135M single-GPU study reports GSM8K exact match of **1.82%** for the base
model, falling to **1.59%** after GRPO-style RLVR at 192-token completions and **1.21%** at
320 tokens; the authors are careful to call it a single-run failure mode rather than a
general bound ([2606.22189](https://arxiv.org/abs/2606.22189), Jun 2026). At `p = 0.0182`
and `G = 8`, **86.3% of prompt groups produce no gradient at all.** A configured batch of
64 prompts is an effective batch of `64 × 0.1367 = 8.7` prompts.

Worse, look at *what* the surviving 13.7% contains. The probability of exactly one success
in eight is `C(8,1)·p·(1−p)^7 = 8 × 0.0182 × 0.9818^7 = 0.1280`, which is
`0.1280 / 0.1367 = 93.7%` of all informative groups. And with the `/std` normalizer, a lone
success in a group of eight receives an advantage of

```
Â_success = (1 − 1/8) / sqrt( (1/8)(7/8) ) = 0.875 / 0.33072 = +2.646  =  √7
Â_failure = (0 − 1/8) / 0.33072            = −0.378              = −1/√7
```

against `±1.000` for a 4-of-8 group. **A lucky single success on a hard prompt receives
2.65× the gradient weight of a genuine success on a prompt the model half-understands.**
At a 1.82% pass rate on GSM8K, a large share of "successes" are the verifier's last-number
fallback matching a coincidentally-correct integer. `[A]` Medium-high confidence that this
is the dominant mechanism behind the reported decrease: the arithmetic is exact, the
attribution to *this* mechanism is inference. Note honestly that it does **not** explain
the reported ordering across completion budgets (320 tokens was worse than 192, not
better); something else is also happening there.

This is also exactly Dr. GRPO's objection to `/std` made concrete: the normalizer makes the
per-group gradient magnitude constant regardless of how much information the group carries,
which upweights the least informative groups the most `[C]`
([2503.20783](https://arxiv.org/abs/2503.20783)).

**The escape hatch, and it is real.** Everything above is a statement about *pass rate*,
not about *parameter count*. Pick a synthetic task with a tunable difficulty knob such that
a 300M policy sits at `p ∈ [0.2, 0.6]` and the collapse fraction drops below 10%. Countdown
is the existence proof `[C]` ([2503.01307](https://arxiv.org/abs/2503.01307)), and
procedural reasoning-task generators generalize it. Every GRPO pathology — collapse,
entropy decay, length inflation with and without `1/|o_i|` — then becomes observable and
matched-budget-able. That is a mechanism programme, and it is the version of RLVR this lab
could actually run.

### 3.9 Reward overoptimization, as a curve

Gao et al. fit gold-reward-vs-proxy-optimization curves parameterized by the *square root*
of the KL divergence from the initial policy, `d = sqrt(KL(π ‖ π_init))`, and find two
functional families: a linear-in-`d` correction for best-of-`n` sampling and a `d·log d`
correction for RL `[C]` ([2210.10760](https://arxiv.org/abs/2210.10760)). The shape is what
matters:

```
proxy reward r_φ:   monotonically increasing in d          (by construction — you optimized it)
gold reward   r*:   rises, peaks, then FALLS               (the thing you wanted)
```

The coefficients scale with reward-model parameter count: bigger reward models peak later
and higher. Two operational reads:

- **`d` is your only knob and your only odometer.** KL from the initial policy is the
  distance travelled, and the peak occurs at a specific distance. `β` in DPO and the KL
  coefficient in PPO/GRPO are both attempts to bound `d` indirectly.
- **You cannot see the peak from inside the run.** The proxy curve is monotone. The peak is
  only visible against a held-out gold signal you evaluate separately. This is the "no miss
  signal" break from §2.3 in its most expensive form.

### 3.10 The rollout group is the most prefix-shareable workload that exists

The crossing point into the memory track, and the one systems analogy in this module that
*does not* break. GRPO samples `G` completions from **one** prompt. The shared prefix is
exact, simultaneous, and known in advance — no hit-rate estimation required.

With a paged KV cache and prefix sharing (`paged-attention-and-prefix-reuse`), the prompt's
KV is stored once instead of `G` times:

```
naive KV tokens   = G · (P + C)
shared KV tokens  = P + G · C
reduction factor  = G(P + C) / (P + G·C)
```

| Symbol | In words |
|---|---|
| `P` | prompt tokens (shared across the group) |
| `C` | completion tokens (unique per group member) |
| `G` | group size |

Limits: as `C → 0` the factor approaches `G` (perfect sharing); as `C → ∞` it approaches 1
(no sharing). Worked:

| `P` | `C` | `G` | naive | shared | reduction |
|---|---|---|---|---|---|
| 512 | 512 | 8 | 8,192 | 4,608 | 1.78× |
| 2048 | 512 | 16 | 40,960 | 10,240 | **4.00×** |
| 512 | 4096 | 8 | 36,864 | 33,280 | 1.11× |

And in bytes, at our ablation scale (`L = 24` layers, `H_kv = 4` KV heads,
`d_head = 64`, bf16), which is `2 · 24 · 4 · 64 · 2 B = 24 KiB/token` from
`research/memory/kv-cache-mechanics.md`:

```
one group, G=8, P=C=512:      4,608 tok × 24 KiB  =  108 MiB
batch of 64 prompts:          64 × 108 MiB        =  6.75 GiB   (shared)
                              64 × 8,192 × 24 KiB =  12.0 GiB   (naive)

batch of 64, G=8, P=512, C=4096:   64 × 33,280 × 24 KiB = 48.8 GiB  (shared)
                                   64 × 36,864 × 24 KiB = 54.0 GiB  (naive)
```

**That last row is the finding.** A long-completion rollout batch — 64 prompts, 8 samples,
4096-token chains of thought — needs **48.8 GiB of KV even with perfect prefix sharing**,
against the `[M]` ≥62 GiB fast tier measured in
`notebook/uma-carveout-controls-fast-tier.md`. It fits, barely, with nothing left for
weights, optimizer state, or activations. The KV cache of an RL rollout batch is the first
thing in a post-training pipeline that touches our capacity ceiling — and it does so at
*generation* time, not training time. This is measurable today with no RL code at all
(exercise C).

---

## 4. Why it matters for Proteus and Mnemosyne

**Four couplings, in descending order of how much they should change what we build.**

### 4.1 The loss interface the memory track needs is the same one DPO needs

`loss-and-optimization` §*Why it matters for Proteus* already requires
`loss.reduction="none"` plus a token mask, because every memory-policy experiment is scored
by a per-token loss delta sliced to an answer span. DPO needs precisely the same primitive:
`log π(y|x) = −Σ_t CE_t` over a response span. So does SFT masking. So does any RLVR
log-probability computation.

**One interface, three consumers, and it must exist before any of them.** If the loss
function only returns a scalar, Mnemosyne's attribution experiments cannot run and no
preference method can be implemented on top of Proteus. Add to the config surface:

| Config field | What it must support |
|---|---|
| `loss.reduction` | `mean` \| `sum` \| `none` — `none` returns a `[B, T]` tensor |
| `loss.token_mask` | a `[B, T]` boolean; also the SFT assistant mask |
| `loss.span_reduce` | `sum` per contiguous span — the operation that turns per-token CE into `log π(y|x)` |
| `loss.implementation` | `default` \| `chunked` \| `fused_linear` — see §3.6's 21-pair fault boundary |

### 4.2 The chat template is a config surface, and reserved token ids are cheap insurance

Two artifact-level lessons from the reference model, both directly actionable for Proteus.

**Reserve special token ids at tokenizer construction time.** In Laguna S 2.1 the thinking
and tool-call tokens are single ids in the *base* vocabulary — `<think>`=18, `</think>`=19,
`<assistant>`=23, `</assistant>`=24, `<tool_call>`=25, `</tool_call>`=26 — and none of them
appear in `added_tokens`, so they were reserved before pretraining rather than grafted on
`[M]` (read from the artifact by `research/notes/posttraining-pipelines.md` §6, revision
`b0a9fd7c850e` per `PROVENANCE.md`). Grafting a token on after pretraining gives it an
untrained embedding row that post-training has to learn from scratch, against a vocabulary
where every other row has billions of tokens of gradient behind it. Reserving 32 ids costs
`32 ids × d × 2 matrices × 2 B` (input embedding plus untied output head) — at `d = 768`,
**96 KiB** — and buys the option to add a post-training
feature that does not exist yet. Take it.

**Version the template and hash it into every run record.** §2.2 point 2 is not
hypothetical: the two copies of the Laguna template in this repository disagree on the
default value of `enable_thinking` and one has a `preserve_thinking` flag the other lacks.
`themis` must record the template hash in the run manifest, exactly as it records the seed.
A run whose mask came from a different template is a different experiment.

### 4.3 The rollout group is a free, high-value Mnemosyne benchmark

§3.10 shows a GRPO rollout batch is the cleanest prefix-sharing workload that exists, and
that at long completion lengths it lands within a factor of ~1.3 of our measured fast-tier
ceiling. **Measuring the KV economics of a rollout group needs no RL, no reward, and no
training** — it needs a generation loop and a KV accountant. That makes it one of the
cheapest experiments available to Mnemosyne, and it produces a number nobody has published:
the resident-KV curve for `(prompts × G × length)` under shared vs unshared prefixes on
unified memory.

The deeper coupling, and it is `research/memory/open-problems-ranked.md` §1 (attribution) in
disguise: **if rollouts are generated under KV eviction or KV quantization, the sampled
distribution shifts, and that shift enters the policy gradient.** A memory decision becomes
a training-signal decision. That is training-inference mismatch by another name `[C]`
([2605.14220](https://arxiv.org/abs/2605.14220), May 2026), and as far as the survey can
find, nobody has measured whether rollout-time KV eviction biases the GRPO update. It is
the direct crossing point of this lab's two tracks.

Symmetrically: alignment behaviour *induced* by post-training can be destroyed by a pure
memory decision. Refusal-rate collapse under KV quantization has been reported at
perplexity deltas too small for PPL-only evaluation to notice `[C]`
([2606.09864](https://arxiv.org/abs/2606.09864), Jun 2026).

### 4.4 Thinking mode changes the cache contract, and three vendors disagree about it

Laguna's model card states it "works best with *preserved thinking*" — keep
`reasoning_content` from prior assistant messages — and "may stop reasoning in follow-up
steps if prior thinking blocks are dropped" `[C]`
(`research/reference/models/laguna-s/README.md:158-161`). Qwen3's published best practice
is the opposite: strip thinking from history, with its template implementing a *rolling*
window `[C]` ([2505.09388](https://arxiv.org/abs/2505.09388) plus its model card, accessed
2026-07-26). MiniMax-M2 states interleaved thinking as a first-class agent modelling
principle `[C]` ([2605.26494](https://arxiv.org/abs/2605.26494), May 2026).

These are three different **cache contracts** hiding behind the same `<think>` tag, and the
difference is invisible in any benchmark table:

| Contract | Prefix-cache behaviour | Capacity behaviour |
|---|---|---|
| preserve-all (Laguna) | history stays append-only → block-hash chains and radix prefixes keep matching | reasoning tokens accumulate across the whole trajectory |
| rolling window (Qwen3) | history is **rewritten** as the window advances → every chained hash from the edit point is invalidated | bounded |
| strip-all | rewritten once per turn | smallest |

For Proteus this is a design decision with a measurable cost on both axes, and it is a
decision the *model card* makes on the serving system's behalf. It is also `[C]` a training
disclosure in disguise: "preserved thinking works best" means the model was optimized on
trajectories where prior reasoning was retained, so dropping it moves the input
off-distribution.

---

## 5. What our register already constrains

Restated from `ASSUMPTIONS.md` rather than re-derived, because every number below bounds
something in section 3:

- `[M]` fast tier **≥62 GiB at ~200 GB/s** (`gpu-fast-tier-size`, single run per arm — an
  anecdote by house standard, but the only number we have). Bounds §3.10's rollout KV.
- `[M]` single tensors **≥32 GiB hang silently at 0% CPU** (`large-tensor-fault-32gib`).
  Bounds §3.6's preference-pair batch at 21 pairs × 2048 tokens.
- `[M]` **20.9 TFLOP/s bf16 GEMM at 8192³** (`gemm-throughput-below-reference`, 63% of the
  cited figure for this silicon). Bounds §3.1's SFT wall-clock.
- `[M]` `F.scaled_dot_product_attention` retains the `B·nh·T²` score matrix by default —
  **147.2 vs 6.6 bytes/T²** with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`
  (`sdpa-is-memory-efficient`). Long-completion rollouts are the exact workload that walks
  `T²` into the 32 GiB fault. `flash_sdp_enabled()` returns `True` either way.
- `[M]` hipBLASLt configuration changes bf16 long-reduction relative error by **~2.8×**
  (2.01e-3 configured vs 5.60e-3 unset, `hipblaslt-config`). It is a **numerics control**,
  not a throughput knob. Every post-training run must record whether it was set.
- `[M]` `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction` is **inert** on
  this stack — toggling it changes the result by exactly zero bits
  (`bf16-reduced-precision-knob-works`, refuted). Do not use it as an experimental axis.
- `[C]` **Distributed collectives are incomplete on gfx1151** (`single-device-only`). No
  disaggregated rollout engine, no actor/learner split, no weight resync over NCCL.
- `[C]` **bf16 numerics are unproven** (`bf16-numerics-unproven`). The Hardware Validation
  Gate has not run. **No number produced on this machine is evidence yet**, including every
  number an exercise below asks you to produce.

---

## 6. Read the code

Paths relative to `research/reference/`. Clones are gitignored; run
`scripts/fetch_reference.sh` first. There is **no DPO, GRPO, PPO, or reward-model trainer
in this reference library** — TRL, verl, and OpenRLHF are not vendored. That is a deliberate
consequence of section 8: we are not going to run those loops, so we read the *primitives*
they compose instead. Everything below is a primitive that a post-training stack is built
out of, and reading them in order is enough to reconstruct what TRL does.

### 6.1 The SFT loss mask, end to end

| Where | What to look at, and why |
|---|---|
| `training/nanogpt/model.py:187`<br>`F.cross_entropy(..., ignore_index=-1)` | **The entire SFT masking mechanism already exists, as one keyword argument.** Set a label to `-1` and that position contributes nothing. Note the value: nanoGPT uses `-1`, HuggingFace's universal convention is `-100`. Copy a `-100` mask into this call and you do not get silent no-masking — you get `IndexError: Target -100 is out of bounds` on CPU, or an asynchronous device-side assert on GPU that surfaces at an unrelated line several kernels later. Know which convention your code is in |
| `architecture/transformers/src/transformers/utils/chat_template_utils.py:431`<br>`class AssistantTracker(Extension)` | The Jinja extension that implements `{% generation %}`. `tags = {"generation"}` at `:433` is the entire registration. This is a *template* extension, so the mask is defined by a file inside the model artifact |
| `.../chat_template_utils.py:452`<br>`start_index = len("".join(self._rendered_blocks))` | The span is recorded as **character** offsets into the rendered string, not token indices. The token conversion happens much later and in a different file, which is where the interesting failures live |
| `.../chat_template_utils.py:508`<br>`if return_assistant_tokens_mask and not re.search(r"\{\%-?\s*generation\s*-?\%\}", chat_template)` | The silent-failure guard: a template without `{% generation %}` produces a `warning_once` and an **empty** span list. Not an exception. This is the "an ACL denies, a mask fails silently" break from §2.2 |
| `.../tokenization_utils_base.py:3152`<br>`start_token = out.char_to_token(i, assistant_start_char)` | Character offsets converted to token indices, per example, in a Python loop. Requires a fast tokenizer — `architecture/transformers/src/transformers/processing_utils.py:2061`<br>`if not is_tokenizers_fast:` raises if you only have a slow one |
| `.../tokenization_utils_base.py:3154`<br>`if start_token is None: break` | **Truncation silently truncates the mask too.** If `max_length` cut the sequence before an assistant span starts, the loop `break`s and every later span is left unmasked-and-untrained. Long agentic trajectories hit this constantly |
| `models/laguna-s/chat_template.jinja:45` and `:77` | `{%- generation -%}` … `{%- endgeneration -%}` — the assistant span in our reference model, spanning from `<assistant>` through the closing `</assistant>\n` |
| `models/laguna-s/chat_template.jinja:58`<br>`{{- '</think>' -}}` | **Inside** the generation span. In thinking-off mode the template emits a bare closing `</think>` with no opener, and because it is inside the span, the model **gets gradient on it**. Thinking-off SFT trains the model to emit an immediate `</think>`; it is not a sampling flag, it is a learned behaviour |
| `models/laguna-s/chat_template.jinja:4` vs `architecture/llama-cpp-laguna/models/templates/poolside-Laguna-S-2.1.jinja:4` | `enable_thinking \| default(true)` in the HF model repo, `default(false)` in the llama.cpp copy. `models/laguna-s/chat_template.jinja:6` also defines a `preserve_thinking` flag the llama.cpp copy does not have, used at `:55`. Two shipped copies, two different rendered training sequences. `generation_config.json:15-17` independently sets `default_chat_template_kwargs: {"enable_thinking": true}` |
| `training/smollm/text/finetuning/train.py:96`<br>`dataset_text_field=args.dataset_text_field` | The shipped SmolLM2 finetuning example does **no masking at all** — it trains full-sequence LM loss on a raw text field (`content`, defaulting to a Python code corpus). It is continued pretraining wearing the name "finetuning." A useful calibration on how much of the published "SFT" surface actually masks |
| `training/smollm/text/finetuning/train.py:110`<br>`optim="paged_adamw_8bit"` | bitsandbytes. `[C]` bitsandbytes crashes on import on gfx1151 (CLAUDE.md). This one line makes the reference SFT script non-runnable here as written; the fix is `adamw_torch` and 4 extra bytes per parameter of optimizer state |

### 6.2 The reward model, which is four lines

| Where | What to look at, and why |
|---|---|
| `architecture/transformers/src/transformers/modeling_layers.py:126`<br>`self.score = nn.Linear(config.get_text_config().hidden_size, self.num_labels, bias=False)` | **The entire reward-model architecture.** With `num_labels=1` this is a `d × 1` matrix replacing the `d × V` LM head. At Laguna's `d = 3072`: 3,072 parameters instead of 308M |
| `.../modeling_layers.py:154`<br>`logits = self.score(hidden_states)` | The score is computed at **every** position and then thrown away except one. Cheap here because the head is one-wide; note the contrast with the LM head, where the same pattern is the largest allocation in the step (`loss-and-optimization` §*Why the logits tensor*) |
| `.../modeling_layers.py:169`<br>`last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)` | The pooling index: the rightmost position whose token is not `pad_token_id`. **Hazard:** the near-universal convention `pad_token = eos_token` makes the real final EOS look like padding, so the score is read one position early. For a reward model whose score is conventionally taken at EOS, that is a systematic off-by-one on every example, and it produces a model that trains fine and scores the wrong thing |
| `.../modeling_layers.py:177`<br>`pooled_logits = logits[torch.arange(batch_size, ...), last_non_pad_token]` | The gather. Everything a reward model *is* lives in lines 126, 154, 169 and 177 |

### 6.3 The verifier — read this before believing an RLVR number

| Where | What to look at, and why |
|---|---|
| `training/lm-evaluation-harness/lm_eval/api/metrics.py:240`<br>`score_list = predictions == references` | **The entire verifiable reward, as one `==`.** Everything else in `exact_match_hf_evaluate` (`:210`) is optional normalization: strip regexes, lowercase, drop punctuation, drop digits |
| `training/lm-evaluation-harness/lm_eval/api/registry.py:456`<br>`"generate_until": ["exact_match"]` | The default metric for free-form generation tasks. This is what "GSM8K exact match" means in every paper table you will read |
| `training/smollm/text/evaluation/smollm2/math_utils.py:209`<br>`def strip_string` | ~140 lines of LaTeX and unit normalization whose only job is making that `==` mean something. Read it as the schema-normalization layer in front of an equality join, and notice that every rule in it is a place two mathematically equal answers can compare unequal |
| `training/smollm/text/evaluation/smollm2/math_utils.py:350`<br>`def extract_answer` | The parser: `\boxed{...}` with a brace-matching walk, then "the answer is", then "final answer is" |
| `training/smollm/text/evaluation/smollm2/math_utils.py:387`<br>`pattern = "-?\d*\.?\d+"` | **The fallback that decides small-model RLVR.** If nothing else parses, take the **last number** in the output. A 300M model that emits no `\boxed{}` is scored on whatever integer it happened to type last. §3.8's "lucky success" is this line |
| `training/smollm/text/evaluation/smollm2/math_utils.py:426`<br>`gt_cot, gt_ans = text.split("####")` | GSM8K ground truth is a string split on `####`. The oracle is a `str.split` |

### 6.4 Rollouts, and where they meet the memory track

| Where | What to look at, and why |
|---|---|
| `training/nanogpt/model.py:306`<br>`def generate` | The rollout loop with **no KV cache** — `:316` re-runs the full prefix on every sampled token, quadratic by construction, and `:314` truncates the left of the sequence at `block_size` with no fault path. This is what an RL rollout costs before anyone optimizes it, and it is the baseline your throughput measurement in exercise C should beat |
| `memory/vllm/vllm/v1/core/kv_cache_utils.py:596`<br>`hash_block_tokens` | The prefix-cache key is a *chain* — `hash(parent_hash, token_ids, extra_keys)` — so it is position-dependent and strictly prefix-ordered. This is precisely why a GRPO group shares perfectly (identical prompt, identical offsets) and why a rolling-window thinking policy (§4.4) invalidates everything downstream of the edit |
| `memory/sglang/python/sglang/srt/mem_cache/radix_cache.py:355`<br>`match_prefix` | The same sharing contract as a radix tree, with `extra_key` namespacing the tree like an ASID. Note that `match_prefix` is a **mutating** read (it may split a node), so "lookup cost" is not read-only |
| `training/olmo-core/src/olmo_core/nn/functional/cross_entropy_loss.py:35`<br>`logits = logits.float()` | The unconditional fp32 upcast that makes §3.6's logits table read 12 bytes per element rather than 4. In DPO you pay it on two sequences per example |
| `training/olmo-core/src/olmo_core/nn/lm_head.py:268`<br>`logits = None` | The fused path where logits are never materialized. This is the escape from the 21-pair fault boundary in §3.6, and `[C]` its numerical equivalence on gfx1151 is unverified — `bf16-numerics-unproven` is open and the accumulator comment at `:278` shows exactness needed defending at least once upstream |

---

## 7. Exercises

All three assume `. .\scripts\activate-lab.ps1` first (native Windows, gfx1151, single
device; the script sets `HIPBLASLT_TENSILE_LIBPATH` and `TORCH_BLAS_PREFER_HIPBLASLT` and
deliberately leaves `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` unset — read the comment at
`scripts/activate-lab.ps1:26-43` before overriding it). Put scratch scripts under
`notebook/`; per house rules they are exempt from TDD only while they stay one-off and
reproducible from committed config.

**Hard safety constraints, from the register.** `[M]` A single 32 GiB tensor hard-hangs
this machine at 0% CPU with no error and requires a force-kill — cap any single allocation
at **24 GiB**. `[M]` `ENVIRONMENT.md` reports **32 GB of system RAM** after the 96 GB UMA
carve-out, so CPU fallbacks must stay small; 8 GiB is a sane host-side cap. `[M]` SDPA
retains the `B·nh·T²` score matrix by default, so exercise C's long-generation sweep is the
one most likely to walk into the fault — keep `T` bounded and watch peak memory.

**None of these have been run by the module author.** They are constructed so that the
predicted value is derivable by hand from section 3, which is what makes the result
checkable rather than merely produced. Where a prediction and your measurement disagree,
the arithmetic above is the thing to attack first.

### Exercise A — Measure the mask fraction of your own SFT data (easy, 30–45 min)

**What you build.** A script that renders conversations through the *actual* Laguna chat
template with `return_assistant_tokens_mask=True`, counts what fraction of tokens receive
gradient, and demonstrates two of the failure modes from §2.2 and §6.1.

This is CPU-only — a tokenizer and a Jinja renderer, no model weights. It runs identically
on the Z13 and on any laptop.

```python
from transformers import AutoTokenizer
import json, pathlib

REF = pathlib.Path("research/reference/models/laguna-s")
tok = AutoTokenizer.from_pretrained(REF)          # reads tokenizer.json + chat_template.jinja

convo = [
    {"role": "user", "content": "Refactor this loop to avoid the allocation."},
    {"role": "assistant", "content": "Hoist the buffer out of the loop body.",
     "reasoning_content": "The allocation is inside the hot path; the buffer is "
                          "the same size every iteration, so it can be reused."},
    {"role": "user", "content": "Now do it without changing the signature."},
    {"role": "assistant", "content": "Use a module-level pool keyed by size."},
]

for thinking in (False, True):
    out = tok.apply_chat_template(
        convo, tokenize=True, return_dict=True,
        return_assistant_tokens_mask=True, enable_thinking=thinking,
    )
    m = out["assistant_masks"]
    print(json.dumps({
        "enable_thinking": thinking,
        "total_tokens": len(m),
        "gradient_tokens": sum(m),
        "mask_fraction": round(sum(m) / len(m), 4),
        "flops_multiplier_vs_naive": round(len(m) / sum(m), 3),
    }))
```

**Then the three checkable claims.**

1. **`μ` and the budget multiplier.** Report `mask_fraction` for both settings. The
   multiplier `1/μ` is the correction from §3.1 — the factor by which your real cost per
   gradient token exceeds `6N`. Do this on 200 examples of whatever data you would actually
   SFT on and report the distribution, not one number.
2. **The `</think>` is inside the gradient span.** With `enable_thinking=False`, decode the
   masked tokens (`tok.convert_ids_to_tokens([i for i, k in zip(out["input_ids"], m) if k])`)
   and confirm token id **19** (`</think>`) is present and masked-in. That is
   `chat_template.jinja:58` sitting inside the `{% generation %}` block at `:45`–`:77`. It
   proves thinking-off is a *trained behaviour*, not a sampling flag.
3. **The silent-mask failure.** Strip the `{%- generation -%}` / `{%- endgeneration -%}`
   lines from a copy of the template, pass it via `chat_template=`, and re-run. Expected: a
   `warning_once` from `chat_template_utils.py:508`, and an all-zero mask. Then compute what
   your loss would be — `sum(mask)` is zero, so a masked mean is `0/0`. Record which of
   `nan`, `0.0`, or an exception your PyTorch produces; that is the shape of the bug you
   would be debugging at 2 a.m.

**Bonus, 5 minutes, and it is the point of the whole exercise.** Hash both shipped
templates (`models/laguna-s/chat_template.jinja` and
`architecture/llama-cpp-laguna/models/templates/poolside-Laguna-S-2.1.jinja`) and confirm
they differ. Then render the same conversation through each with default kwargs and diff
the token ids. The delta is what §4.2 says `themis` must record in the run manifest.

**Runtime.** `[A]` 30–45 minutes including reading the template. No GPU. No CPU fallback
needed — this *is* the CPU path.

### Exercise B — Implement DPO from scratch and catch the chosen log-prob falling (medium, 1.5–2.5 h)

**What you build.** A `dpo_loss` function, a numerical check of the gradient identity from
§3.4, and a small training run that reproduces the likelihood-displacement phenomenon.

Use the nanoGPT `shakespeare_char` checkpoint from the Hardware Validation Gate as both
`π_ref` (frozen) and `π_θ` (initialization). Build synthetic preference pairs by sampling
two completions from the base model for the same 64-token prompt and labelling the one with
more occurrences of a chosen rare character as `y_w`. This is a deliberately silly
preference, and that is fine — the point is the *dynamics*, not the alignment.

```python
import torch, torch.nn.functional as F

def seq_logprob(model, ids, prompt_len):
    """Sum of log p(token_t | prefix) over the RESPONSE span only. Shape: [B]."""
    logits, _ = model(ids[:, :-1])                       # [B, T-1, V]
    lp = torch.log_softmax(logits.float(), dim=-1)
    tgt = ids[:, 1:]                                     # [B, T-1]
    per_tok = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    mask = torch.zeros_like(per_tok, dtype=torch.bool)
    mask[:, prompt_len - 1:] = True                      # response tokens only
    return (per_tok * mask).sum(-1)

def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    u = beta * ((pi_w - ref_w) - (pi_l - ref_l))         # the margin
    return -F.logsigmoid(u).mean(), u.detach()
```

**Three checkable numbers.**

1. **The gradient identity, to 1e-6.** From §3.4 the gradient weight on
   `∇log π(y_w) − ∇log π(y_l)` is exactly `−β·σ(−u)`. Verify it: take one pair, compute
   `loss.backward()`, and separately compute
   `w = -beta * torch.sigmoid(-u)`; then confirm `∂L/∂(pi_w) == w` and
   `∂L/∂(pi_l) == -w` by autograd on the scalar log-probs
   (`torch.autograd.grad(loss, [pi_w, pi_l])`). This is a pure math check and must pass
   on CPU and GPU identically. **If it does not agree to 1e-6 in fp32, stop** — you have
   found something about this build that matters more than DPO.
2. **Both log-probs fall.** Train 200 steps at `β = 0.1`, `lr = 5e-7` (DPO learning rates
   are 100–1000× smaller than SFT's; at SFT rates it diverges immediately). Log four series
   per step: `mean(pi_w)`, `mean(pi_l)`, the margin `u`, and the implicit-reward accuracy
   `mean(u > 0)`. Prediction from §3.4(a): **the margin rises monotonically while
   `mean(pi_w)` also falls.** Report the sign of `Δ mean(pi_w)` over the run. If it rises,
   you have a case where the shared-structure argument does not bind — write that up, it is
   a real finding at this scale and the literature does not have a small-model
   characterization of when displacement does and does not occur.
3. **Where the mass went.** Before and after, sample 64 completions from the same prompts
   and measure the fraction that match neither `y_w` nor `y_l` under a loose string
   criterion, plus the mean entropy of the next-token distribution. §3.4(b) predicts mass
   moves to sequences in neither. Quantify it.

**Then the cheap and useful variant.** Re-run with SimPO's loss (§3.5) — no reference
model, length-normalized, `γ = 0.5` — and compare the mean chosen-response length
trajectory. Prediction: DPO's grows and SimPO's does not. That is §3.5 made visible in
about 15 extra minutes.

**Runtime.** `[A]` 1.5–2.5 h total, of which ~30 min is the two 200-step runs on GPU. Low
confidence on the GPU time: nobody has measured this recipe on this machine. Set
`compile=False` on the first attempt; `[A]` `torch.compile` on gfx1151 is unproven and a
compile failure is indistinguishable from a loss bug.

**CPU fallback.** nanoGPT's published CPU configuration — 4 layers, 128 channels,
`block_size` 64 (`training/nanogpt/README.md:85`) — with 100 steps instead of 200. `[A]`
40–70 minutes. The gradient identity check (claim 1) is device-agnostic and takes seconds
either way; run it first regardless of which path you take.

### Exercise C — Price the GRPO run you are not going to do (medium, 1–2 h)

**What you build.** The honest cost model for RLVR on this machine, in three parts: a
closed-form collapse table you verify by Monte Carlo, a **measured** batched-decode
throughput number that replaces the `[A]` assumption the survey's wall-clock estimate rests
on, and the resident-KV curve for a rollout batch.

**Part 1 — the collapse table (2 minutes, exact).** No model needed.

```python
import numpy as np
rng = np.random.default_rng(1337)
for p in [0.0182, 0.05, 0.10, 0.25, 0.50]:
    for G in [4, 8, 16, 64]:
        closed = 1 - p**G - (1-p)**G
        k = rng.binomial(G, p, size=1_000_000)
        mc = float(((k > 0) & (k < G)).mean())
        print(f"p={p:<7} G={G:<3} closed={closed:.4f} mc={mc:.4f} tax={1/closed:.2f}x")
```

Checkable: `closed` and `mc` must agree to ~3 decimals. Confirm the `p=0.0182, G=8` row
reads **0.1367** and a **7.32×** rollout tax — that is §3.8's central number, reproduced
from scratch.

**Part 2 — measured decode throughput (the valuable half).** The survey's RLVR wall-clock
estimate rests on an explicitly-flagged `[A]` assumption of 2,000 tok/s aggregate
(`research/notes/posttraining-pipelines.md` §8, medium confidence). Replace it. Instantiate
a model at our ablation scale — `[A]` 24 layers, `d = 768`, 12 heads, 4 KV heads,
`d_head = 64`, which is roughly 300M with a small vocabulary — **weights can be randomly
initialized**, since throughput does not depend on training. Generate with a real KV cache
(not nanoGPT's cacheless `generate`) at the rollout batch shapes:

| batch (prompts × G) | prompt tokens | new tokens | what it tells you |
|---|---|---|---|
| 8 × 8 = 64 | 512 | 512 | small-group throughput |
| 32 × 8 = 256 | 512 | 512 | does throughput scale with batch, or is it bandwidth-capped? |
| 64 × 8 = 512 | 512 | 512 | the shape the survey assumed |
| 16 × 8 = 128 | 512 | 2048 | long-chain regime |

Record aggregate tokens/s, `torch.cuda.max_memory_allocated()`, and the wall-clock. Cap
runs so no single tensor approaches 24 GiB, and record whether
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` was set — `[M]` it changes attention activation
memory by ~18× and, being experimental, it changes numerics
(`ASSUMPTIONS.md: sdpa-is-memory-efficient`). Run each configuration **in a fresh process**
and at least twice; a single observation is an anecdote.

**Part 3 — combine.** With `R` = measured aggregate tok/s:

```
tokens per step   = prompts × G × completion_length
steps             = 1000
naive tokens      = 64 × 8 × 512 × 1000              = 262,144,000
with dynamic sampling at p=0.0182:  × 7.32           = 1,918,894,080
wall clock        = tokens / R
```

At the survey's assumed `R = 2000` that is **266 hours ≈ 11 days** for a run whose published
analogue at 135M made the metric *worse*. Report the number your machine actually gives.
**Whatever `R` you measure, this is the exercise's deliverable: a defensible sentence of the
form "GRPO on GSM8K at our scale costs N days and the literature predicts it will not
work,"** which is worth more than an RL implementation.

**Part 4, optional (20 min) — the memory-track crossing.** Compute the resident KV for each
row above under both the naive (`G·(P+C)`) and prefix-shared (`P + G·C`) formulas from
§3.10 at 24 KiB/token, and check the shared numbers against your measured
`max_memory_allocated()`. If your generation loop is not prefix-sharing (it almost certainly
is not), the gap you measure is exactly the value Mnemosyne would add to a rollout engine —
`[A]` predicted at 1.8× for the 512/512/8 shape and 4.0× at 2048/512/16.

**Runtime.** `[A]` 1–2 h. Part 1 is 2 minutes anywhere. Part 2 is the GPU-bound half;
`[A]` 30–60 min for four configurations run twice. **CPU fallback:** run part 2 at
`d = 256`, 6 layers, batch 8, 128 new tokens, and report tok/s — the absolute number is
meaningless for planning, but the *scaling shape* across batch sizes is the thing you are
looking at, and it is visible on CPU. `[A]` 20–30 min.

---

## 8. What is still unsolved here

Not "future work" — the places where the map ends, separated into what the field does not
know and what we cannot do.

### 8.1 Contested in the literature, and left contested

- **Does RLVR create capability or only sharpen sampling?** Base models match or beat
  RLVR-trained models at pass@k for large k, implying reweighting within existing support
  `[C]` ([2504.13837](https://arxiv.org/abs/2504.13837)); ProRL reports genuinely expanded
  boundaries under long-horizon RL `[C]`
  ([2505.24864](https://arxiv.org/abs/2505.24864)); another line argues RLVR incentivizes
  correct *reasoning*, not just correct answers `[C]`
  ([2506.14245](https://arxiv.org/abs/2506.14245)); and "Spurious Rewards" reports
  substantial gains on Qwen models from *random or wrong* rewards, which implicates the base
  model rather than the signal `[C]`
  ([2506.10947](https://arxiv.org/abs/2506.10947)). Unresolved. The disagreement is largely
  about which base model and how long you train — both axes, not confounds.
- **SFT memorizes and RL generalizes, or the reverse?** `[C]`
  ([2501.17161](https://arxiv.org/abs/2501.17161)) versus `[C]`
  ([2509.21128](https://arxiv.org/abs/2509.21128)), which reports the opposite
  decomposition: RL narrows the output distribution while SFT broadens it. Both measure
  real things and disagree about which one is called generalization.
- **Is the GRPO normalization correct?** Dr. GRPO says `/std` and `1/|o_i|` are biased
  `[C]` ([2503.20783](https://arxiv.org/abs/2503.20783)); most implementations keep them.
- **Is training-inference mismatch a precision bug or an optimization problem?** `[C]`
  ([2605.14220](https://arxiv.org/abs/2605.14220), May 2026) versus `[C]`
  ([2602.01826](https://arxiv.org/abs/2602.01826), Feb 2026).
- **Is offline preference optimization still a required stage?** §2.4. **No matched-budget
  ablation exists.**
- **Preserved vs rolling-window vs stripped thinking across turns.** §4.4. No public
  head-to-head on either accuracy or cache cost.
- **Do rubric rewards extend RLVR or just relocate the hack?** `[C]`
  ([2507.17746](https://arxiv.org/abs/2507.17746)) versus `[C]`
  ([2605.12474](https://arxiv.org/abs/2605.12474), May 2026).

### 8.2 What we cannot do on this hardware, stated plainly

- **Agentic RL. Not at all.** Multi-turn rollouts against a live environment need
  environment infrastructure, trajectory lengths in the tens of thousands of tokens, and
  wall-clock we do not have. Read the credit-assignment survey `[C]`
  ([2604.09459](https://arxiv.org/abs/2604.09459), Apr 2026); do not build a sandbox.
- **Any disaggregated RL topology.** `[C]` Collectives are incomplete on gfx1151
  (`single-device-only`), so there is no separate rollout engine, no actor/learner split,
  and no weight resync over NCCL. Colocated single-device RL is possible in principle, but
  every published recipe assumes the disaggregated shape, so reproducing one means
  re-engineering it first `[C]`
  ([2409.19256](https://arxiv.org/abs/2409.19256),
  [2505.24298](https://arxiv.org/abs/2505.24298)).
- **RLVR on human benchmarks at 20M–300M.** §3.8's arithmetic plus the published 135M
  result `[C]` ([2606.22189](https://arxiv.org/abs/2606.22189)). The wider small-model
  literature agrees on the direction: distillation SFT beats direct RL below about a billion
  parameters `[C]` ([2509.24945](https://arxiv.org/abs/2509.24945),
  [2505.21067](https://arxiv.org/abs/2505.21067),
  [2606.04466](https://arxiv.org/abs/2606.04466), Jun 2026).
- **Any claim about alignment *quality*.** A 300M model has few preferences worth having,
  and we have no human labellers and no judge model we can host that is not also the thing
  under test. We can reproduce preference-optimization *pathologies*; we cannot make an
  alignment claim, and any that appears in this lab's output should be treated as a harness
  bug.
- **Any claim about RL *scaling*.** That has its own predictive-fitting literature at
  compute levels we will never touch `[C]`
  ([2510.13786](https://arxiv.org/abs/2510.13786)).

### 8.3 What we can do that nobody has published

Each of these is small-scale-shaped, and each can fail:

1. **At what pass rate does GRPO stop producing gradient, empirically?** §3.8 gives the
   closed form for an idealized binary verifier with a per-prompt success probability. Real
   pass rates are heterogeneous across prompts and correlated within a group. Sweep
   synthetic-task difficulty so a 300M policy sits at `p ∈ {2, 10, 25, 50, 75}%` and measure
   the actual zero-advantage fraction against the prediction. Falsifies "RL needs scale" by
   replacing it with a measurable threshold.
2. **Does removing `/std` and `1/|o_i|` change the conclusion at our scale, or only the
   length?** Direct replication of a contested claim in a regime nobody has tested `[C]`
   ([2503.20783](https://arxiv.org/abs/2503.20783)). §3.7's length-inflation derivation is a
   prediction this would test.
3. **Is training-inference mismatch observable on gfx1151, and how large?** Generate with
   one code path, score with another, measure per-token log-prob disagreement under bf16 vs
   fp32 on identical weights. **This needs no RL at all**, it feeds a blocking row
   (`bf16-numerics-unproven`), and given `[M]` that hipBLASLt configuration alone moves
   bf16 long-reduction error by 2.8×, it has a live candidate cause before you start.
4. **Does DPO between SFT and RLVR contribute anything under a matched token budget?** The
   ablation nobody published: three arms, SFT→RLVR, SFT→DPO→RLVR, SFT→DPO, on one synthetic
   verifiable task.
5. **Does KV eviction during rollout generation bias the policy gradient?** Identical GRPO
   with full-KV rollouts vs compressed rollouts, measuring divergence in the *update*, not
   in downstream accuracy. §4.3. The direct crossing point of this lab's two tracks and, as
   far as the survey can find, unpublished.

### 8.4 And locally: nothing here is measured yet

The Hardware Validation Gate has not run. `bf16-numerics-unproven` is open, and the
operations this module depends on — log-softmax over 100k classes, a length-`|y|` sum of
log-probabilities, and a logistic loss on their difference — are exactly the long-reduction
shapes where `[M]` hipBLASLt configuration already moves relative error by 2.8×. **No DPO
margin, no reward score, and no pass rate produced on this machine is evidence until that
gate is green.** Section 7's exercises are instructive regardless; their outputs are
provisional.

---

## 9. Self-check

1. Your SFT dataset renders to 4096-token packed sequences with a mask fraction of 0.28.
   You budget "500M tokens" of SFT on a 300M model. How many gradient tokens is that, what
   is the FLOPs cost per gradient token, and roughly how long does it take at the measured
   20.9 TFLOP/s bf16 peak assuming a generous 30% of peak?

2. A DPO run reports: loss falling smoothly, implicit-reward accuracy rising from 0.51 to
   0.88, and `mean log π_θ(y_w|x)` falling from −140 to −190. Is the run broken? What
   quantity is the loss actually optimizing, and what single additional metric would tell
   you whether the model has become worse?

3. Two preference pairs are identical except that in pair A the chosen and rejected
   responses are both 200 tokens, and in pair B the chosen is 800 tokens and the rejected is
   200. The policy's average per-token log-ratio against the reference is +0.008 nats on
   both. With `β = 0.1`, how much implicit-reward margin does pair B get purely from length,
   and what would SimPO's normalization do to it?

4. You are planning a GRPO run with `G = 16` on a task where your 300M policy passes 8% of
   the time. What fraction of prompt groups produce zero gradient, what is the dynamic-
   sampling rollout tax, and if you need 1,000 optimizer steps at 32 prompts × 16 samples ×
   512 completion tokens, how many tokens must you generate?

5. A colleague reports "our reward model's mean score on the eval set went from 1.4 to 3.1
   after finetuning, so the policy improved." Give two independent reasons this sentence
   carries no information, one from §3.2 and one from §3.9.

6. You want the KV-capacity number for a rollout batch: 48 prompts, `G = 8`, 1024-token
   prompts, 3072-token completions, at 24 KiB per token. Compute resident KV both naive and
   prefix-shared, state the reduction factor, and say which of those two numbers fits inside
   the measured fast tier.

---

## 10. Answers

**1.** Gradient tokens: `500e6 × 0.28 = 140M`. Cost per gradient token: `6N/μ =
6 × 3e8 / 0.28 = 6.43e9` FLOPs. Total: `6 × 3e8 × 5e8 = 9.0e17` FLOPs (the mask does not
reduce compute — you process all 500M tokens either way; §3.1). At 30% of the `[M]` 20.9
TFLOP/s peak, i.e. 6.27 TFLOP/s: `9.0e17 / 6.27e12 = 143,540 s ≈ 39.9 hours`. Roughly
**1.7 days for 140M gradient tokens**, which is the number to state when someone proposes an
SFT ablation sweep. Note the honest tag: 20.9 TFLOP/s is `[M]`, the 30% is `[A]` and
unvalidated — the Hardware Validation Gate has not run a known-good recipe here.

**2.** The run is not broken; it is doing exactly what the loss asks. From §3.4(a) the DPO
loss is a function of the *margin* `β(Δ_w − Δ_l)` only, and nothing in it anchors the
absolute level of either term. Accuracy rising to 0.88 means the margin is widening. The
chosen log-probability falling by 50 nats is likelihood displacement: pushing down on `y_l`
pushes down on structure `y_w` shares. What is not visible in these three numbers is *where
the displaced mass went* (§3.4(b)) — it is redistributed over sequences in neither
response. **The single most informative additional metric is `mean log π_θ(y_l|x)` alongside
`mean log π_θ(y_w|x)`**: if both fell and the gap widened, the model is reallocating mass
off-dataset; if `y_l` fell far more, the intended thing is happening. A close second is mean
generated response length (§3.5), because a widening margin driven by length is not a
preference improvement.

**3.** From §3.5, `Δ ≈ |y| · δ̄`. Pair A: `Δ_w − Δ_l ≈ (200 − 200) × 0.008 = 0`, so length
contributes nothing. Pair B: `(800 − 200) × 0.008 = 4.8` nats of log-ratio difference, times
`β = 0.1`, gives **+0.48 of implicit-reward margin from length alone** — and `σ(0.48) =
0.618`, so pair B looks 62% "already correct" to the loss purely because the chosen response
is four times longer. SimPO divides each side by its own length before comparing, so the
implicit rewards become `(β/800)·log π(y_w)` and `(β/200)·log π(y_l)`; the systematic
length term cancels to first order and only the per-token quality difference remains. That
is the entire motivation for the normalization.

**4.** `p = 0.08`, `G = 16`. `p^G = 0.08^16 ≈ 2.8e-18` (negligible).
`(1−p)^G = 0.92^16 = exp(16 × ln 0.92) = exp(16 × −0.083382) = exp(−1.33411) = 0.2634`.
So **26.3% of groups produce zero gradient**, 73.7% contribute, and the rollout tax is
`1/0.7366 = 1.36×`. Tokens: naive `1000 × 32 × 16 × 512 = 262,144,000`; with dynamic
sampling `× 1.36 = 356.4M` generated tokens. Note how much friendlier 8% is than 1.82% —
the tax fell from 7.32× to 1.36×, and the difference is entirely the base model's pass rate,
not its parameter count. That is §3.8's escape hatch stated as a number.

**5.** *From §3.2:* Bradley–Terry constrains only reward *differences* within a prompt, so
`r_φ` is identified only up to an arbitrary per-prompt additive constant. A "mean score"
averages quantities with independent, unconstrained offsets; it can move because the prompt
mix changed, or because nothing changed and the offsets drifted. Only within-prompt
differences are meaningful. *From §3.9:* even if the score were comparable, the proxy reward
is **monotonically increasing in optimization pressure by construction** — you trained the
policy to maximize it. Gold reward rises, peaks, then falls while the proxy keeps climbing,
so a rising proxy score is equally consistent with genuine improvement and with
overoptimization past the peak. The proxy going up is the one observation that cannot
distinguish the two cases.

**6.** `P = 1024`, `C = 3072`, `G = 8`, 48 prompts, 24 KiB/token.

```
naive per group  = G(P + C)   = 8 × 4096      = 32,768 tokens
shared per group = P + G·C    = 1024 + 24,576 = 25,600 tokens
reduction factor = 32,768 / 25,600            = 1.28×

naive total  = 48 × 32,768 × 24 KiB = 37,748,736 KiB = 36.0 GiB
shared total = 48 × 25,600 × 24 KiB = 29,491,200 KiB = 28.1 GiB
```

Both fit inside the `[M]` ≥62 GiB fast tier, but neither leaves room for a 300M policy plus
optimizer state plus a reference copy plus activations if you also intend to train in the
same process — 5.4 GB of model state (§3.7) on top of 36.0 GiB is comfortable, on top of
that *plus* long-completion activations is not. The reduction factor is only 1.28× because
completions are three times the prompt length: prefix sharing pays in proportion to `P/C`
(§3.10), and a long chain of thought is precisely the regime where it pays least. That is
the honest read — prefix sharing is a large win for short-completion rollouts and a rounding
error for reasoning traces.

---

## 11. Sources

`[C]` arXiv ids below are taken from `research/notes/posttraining-pipelines.md`, whose
author resolved every one against the live arXiv API on 2026-07-26, plus
`research/notes/evaluation-landscape.md` and `curriculum/loss-and-optimization.md`.
Resolution proves the paper exists, not that it supports the claim beside it.

**Foundations.** [1707.06347](https://arxiv.org/abs/1707.06347) — PPO ·
[2009.01325](https://arxiv.org/abs/2009.01325) — Learning to summarize from human feedback ·
[2203.02155](https://arxiv.org/abs/2203.02155) — InstructGPT ·
[2005.14165](https://arxiv.org/abs/2005.14165) — Language Models are Few-Shot Learners ·
[2210.10760](https://arxiv.org/abs/2210.10760) — Scaling Laws for Reward Model
Overoptimization.

**SFT and data.** [2305.11206](https://arxiv.org/abs/2305.11206) — LIMA ·
[2501.12948](https://arxiv.org/abs/2501.12948) — DeepSeek-R1 (reasoning-trace distillation) ·
[2411.15124](https://arxiv.org/abs/2411.15124) — Tulu 3 ·
[2512.13961](https://arxiv.org/abs/2512.13961) — Olmo 3 ·
[2509.04259](https://arxiv.org/abs/2509.04259) — RL's Razor.

**Preference optimization.** [2305.18290](https://arxiv.org/abs/2305.18290) — DPO ·
[2310.12036](https://arxiv.org/abs/2310.12036) — IPO ·
[2402.01306](https://arxiv.org/abs/2402.01306) — KTO ·
[2403.07691](https://arxiv.org/abs/2403.07691) — ORPO ·
[2405.14734](https://arxiv.org/abs/2405.14734) — SimPO ·
[2401.08417](https://arxiv.org/abs/2401.08417) — CPO ·
[2404.19733](https://arxiv.org/abs/2404.19733),
[2402.04792](https://arxiv.org/abs/2402.04792),
[2401.10020](https://arxiv.org/abs/2401.10020) — iterative/online DPO ·
[2506.21495](https://arxiv.org/abs/2506.21495),
[2405.19320](https://arxiv.org/abs/2405.19320) — off-policy support limits ·
[2310.03716](https://arxiv.org/abs/2310.03716) — length bias ·
[2402.07319](https://arxiv.org/abs/2402.07319) — ODIN.

**RLVR, GRPO and variants.** [2402.03300](https://arxiv.org/abs/2402.03300) — DeepSeekMath
(GRPO) · [2501.12599](https://arxiv.org/abs/2501.12599) — Kimi k1.5 ·
[2503.14476](https://arxiv.org/abs/2503.14476) — DAPO ·
[2503.20783](https://arxiv.org/abs/2503.20783) — Dr. GRPO ·
[2507.18071](https://arxiv.org/abs/2507.18071) — GSPO ·
[2512.01374](https://arxiv.org/abs/2512.01374),
[2606.12370](https://arxiv.org/abs/2606.12370) — entropy collapse ·
[2605.14220](https://arxiv.org/abs/2605.14220),
[2602.01826](https://arxiv.org/abs/2602.01826) — training-inference mismatch ·
[2510.13786](https://arxiv.org/abs/2510.13786) — RL scaling ·
[2409.19256](https://arxiv.org/abs/2409.19256),
[2505.24298](https://arxiv.org/abs/2505.24298) — disaggregated RL infrastructure.

**Verifiers and their failure modes.**
[2604.15149](https://arxiv.org/abs/2604.15149) — RLVR abandons rule induction ·
[2606.01066](https://arxiv.org/abs/2606.01066) — verifier fuzzing ·
[2510.00915](https://arxiv.org/abs/2510.00915),
[2603.16140](https://arxiv.org/abs/2603.16140) — noisy verifiers ·
[2501.07301](https://arxiv.org/abs/2501.07301) — Qwen's PRM post-mortem ·
[2305.20050](https://arxiv.org/abs/2305.20050) — Let's Verify Step by Step ·
[2507.17746](https://arxiv.org/abs/2507.17746),
[2605.12474](https://arxiv.org/abs/2605.12474) — rubric rewards, contested.

**Does RL create capability?** [2504.13837](https://arxiv.org/abs/2504.13837) ·
[2505.24864](https://arxiv.org/abs/2505.24864) ·
[2506.14245](https://arxiv.org/abs/2506.14245) ·
[2506.10947](https://arxiv.org/abs/2506.10947) ·
[2501.17161](https://arxiv.org/abs/2501.17161) ·
[2509.21128](https://arxiv.org/abs/2509.21128).

**Reasoning behaviour and thinking mode.**
[2503.01307](https://arxiv.org/abs/2503.01307) — cognitive behaviours (Countdown) ·
[2409.12917](https://arxiv.org/abs/2409.12917) — SCoRe ·
[2505.09388](https://arxiv.org/abs/2505.09388) — Qwen3 ·
[2605.26494](https://arxiv.org/abs/2605.26494) — MiniMax-M2 interleaved thinking.

**Small scale, and the memory crossing.**
[2606.22189](https://arxiv.org/abs/2606.22189) — the 135M single-GPU GRPO study ·
[2509.24945](https://arxiv.org/abs/2509.24945),
[2505.21067](https://arxiv.org/abs/2505.21067),
[2606.04466](https://arxiv.org/abs/2606.04466) — distillation beats RL below ~1B ·
[2504.15777](https://arxiv.org/abs/2504.15777) — LoRA reasoning RL at 1.5B for ~$10 ·
[2606.09864](https://arxiv.org/abs/2606.09864) — alignment collapse under KV quantization ·
[2607.08032](https://arxiv.org/abs/2607.08032) — the rate-distortion framing of memory
under a budget ·
[2604.09459](https://arxiv.org/abs/2604.09459) — agentic credit assignment survey.

**Local and measured.** `ASSUMPTIONS.md` — `gpu-fast-tier-size` (≥62 GiB at ~200 GB/s),
`large-tensor-fault-32gib` (silent hang at 0% CPU), `gemm-throughput-below-reference`
(20.9 TFLOP/s bf16 at 8192³), `hipblaslt-config` (2.8× bf16 long-reduction error, a numerics
control), `bf16-reduced-precision-knob-works` (refuted — inert),
`sdpa-is-memory-efficient` (refuted by default; 147.2 vs 6.6 bytes/T²),
`single-device-only`, `bf16-numerics-unproven`, `kv-per-token-laguna`, `reference-model`.
`ENVIRONMENT.md` (2026-07-26; 32 GB system RAM after the 96 GB carve-out, torch
`2.12.0a0+rocm7.13.0a20260313`, HIP 7.2.0). `notebook/uma-carveout-controls-fast-tier.md`.
`scripts/activate-lab.ps1:26-43`. `research/notes/posttraining-pipelines.md` (the survey
this module teaches), `research/notes/evaluation-landscape.md`,
`research/memory/kv-cache-mechanics.md`, `research/memory/open-problems-ranked.md`,
`research/reference/CODE_MAP.md`. `research/reference/models/laguna-s/` — `chat_template.jinja`,
`generation_config.json`, `README.md` (revision `b0a9fd7c850e` per `PROVENANCE.md`).
