#!/usr/bin/env bash
# Rebuild research/reference/ from upstream. THIS SCRIPT is committed; the clones are
# not (see .gitignore). Running it reproduces the reference library from scratch.
#
# Every source below was verified to exist before being listed -- repo, default branch,
# and SPDX license pulled from the GitHub and HuggingFace APIs on 2026-07-26. Nothing
# is here on recall.
#
# Model repos are cloned with GIT_LFS_SKIP_SMUDGE=1: we want configs, tokenizers, chat
# templates and modeling code. No weights, ever (CLAUDE.md hard rule 5).
#
# Usage:
#   scripts/fetch_reference.sh              # fetch everything missing
#   scripts/fetch_reference.sh memory       # fetch one category
#   scripts/fetch_reference.sh --provenance # rewrite PROVENANCE.md from what is on disk
#
# Categories: architecture models training memory hardware

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF_DIR="$REPO_ROOT/research/reference"
PROVENANCE="$REF_DIR/PROVENANCE.md"

# name|category|url|ref|purpose
# ref is a branch name; the SHA actually fetched is recorded in PROVENANCE.md, which is
# what makes a result reproducible. "this stack moves weekly."
MANIFEST='
llama-cpp-laguna|architecture|https://github.com/poolsideai/llama.cpp|laguna|Reference implementation of mixed SWA/global attention, sigmoid MoE gating, FP8 KV cache, DFlash speculative decoding
transformers|architecture|https://github.com/huggingface/transformers|main|The readable Python reference; src/transformers/models/laguna/ is the primary read
mamba|architecture|https://github.com/state-spaces/mamba|main|Mamba-2 selective scan -- the canonical constant-state implementation
samba|architecture|https://github.com/microsoft/Samba|main|Inter-layer hybrid: Mamba + sliding-window attention
zamba2|architecture|https://github.com/Zyphra/Zamba2|main|Inter-layer hybrid with shared attention blocks
megatron-lm|architecture|https://github.com/NVIDIA/Megatron-LM|main|Nemotron-H hybrid configs and the reference parallelism implementations
qwen3|architecture|https://github.com/QwenLM/Qwen3|main|Qwen3-Next / Gated DeltaNet lineage
kimi-linear|architecture|https://github.com/MoonshotAI/Kimi-Linear|master|Kimi Linear attention: hybrid ratio and update rule
minimax-01|architecture|https://github.com/MiniMax-AI/MiniMax-01|main|Lightning Attention at scale
rwkv-lm|architecture|https://github.com/BlinkDL/RWKV-LM|main|RWKV-7 recurrent formulation
xlstm|architecture|https://github.com/NX-AI/xlstm|main|mLSTM/sLSTM matrix-memory recurrence
hymba|architecture|https://github.com/NVlabs/hymba|main|Intra-layer (head-wise) hybrid -- the alternative to layer interleaving
gpt-oss|architecture|https://github.com/openai/gpt-oss|main|SWA+global interleaving in a shipped open model
flash-linear-attention|architecture|https://github.com/fla-org/flash-linear-attention|main|Efficient kernels for most linear-attention variants; the practical reference
laguna-s|models|https://huggingface.co/poolside/Laguna-S-2.1|main|REFERENCE MODEL: config, tokenizer, chat template, model card. No weights
laguna-xs|models|https://huggingface.co/poolside/Laguna-XS-2.1|main|Small sibling -- the one that can actually be run locally
nemotron-nano|models|https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2|main|Nemotron-H hybrid config
qwen3-next|models|https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct|main|Gated DeltaNet config and hybrid ratio
kimi-linear-model|models|https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct|main|Kimi Linear config as shipped
gpt-oss-20b|models|https://huggingface.co/openai/gpt-oss-20b|main|SWA+global config as shipped
olmo|training|https://github.com/allenai/OLMo|main|Fully open training stack: loop, FSDP, dataloader, checkpointer
olmo-core|training|https://github.com/allenai/OLMo-core|main|The current OLMo trainer internals
dolma|training|https://github.com/allenai/dolma|main|Corpus construction, filtering and dedup tooling
torchtitan|training|https://github.com/pytorch/torchtitan|main|PyTorch-native parallelism reference
smollm|training|https://github.com/huggingface/smollm|main|Small-model recipes at our ablation scale
nanogpt|training|https://github.com/karpathy/nanoGPT|master|The known-good tiny recipe for the Hardware Validation Gate
lm-evaluation-harness|training|https://github.com/EleutherAI/lm-evaluation-harness|main|Standard eval harness
vllm|memory|https://github.com/vllm-project/vllm|main|PagedAttention block tables, prefix caching -- the memory-hierarchy reference
sglang|memory|https://github.com/sgl-project/sglang|main|RadixAttention prefix reuse
flashinfer|memory|https://github.com/flashinfer-ai/flashinfer|main|Attention kernels and KV layouts underneath the servers
mooncake|memory|https://github.com/kvcache-ai/Mooncake|main|KV-centric disaggregated serving; prefill/decode split and KV tiering
letta|memory|https://github.com/letta-ai/letta|main|MemGPT lineage: agent memory tiers and compaction
a-mem|memory|https://github.com/agiresearch/A-mem|main|Agentic memory with dynamic organisation
kv-cache-compression-index|memory|https://github.com/October2001/Awesome-KV-Cache-Compression|main|Paper index for KV compression/eviction
agent-memory-index|memory|https://github.com/Shichun-Liu/Agent-Memory-Paper-List|main|Paper index for agent memory
amdsense|hardware|https://github.com/bkpaine1/amdsense|master|93+ ML training experiments on gfx1151; bf16 bugs and reproduction scripts
wsl-rocm|hardware|https://github.com/andweng/wsl-rocm|main|Community ROCm-under-WSL2 setup notes (the path we did NOT take)
rocm|hardware|https://github.com/ROCm/ROCm|develop|Upstream ROCm: issue history and release notes for the stack we pin against
'

# Deliberately NOT in the manifest -- both are gated, and neither is load-bearing:
#
#   google/gemma-3-4b-it            gated: manual approval
#   ai21labs/AI21-Jamba-Mini-1.7    gated: must accept terms
#
# A gated source makes this script un-runnable for anyone without a HuggingFace
# account, and both are redundant. Gemma 3's SWA+global interleaving is covered by
# gpt-oss-20b (ungated, same pattern) and by transformers' configuration_gemma3.py,
# which we clone. Jamba's hybrid ratio is in transformers' configuration_jamba.py and
# in its paper. If an experiment ever needs the exact shipped values, authenticate
# deliberately and add them then -- do not add them to make a table look complete.

# SPDX id as reported by the GitHub / HuggingFace API on 2026-07-26, recorded here so
# PROVENANCE states the actual licence rather than just "a LICENSE file exists".
# NONE = no licence detected: default copyright, so read-only. Never vendored, never
# redistributed. NOASSERTION = a licence file GitHub could not classify; read it.
LICENSES='
llama-cpp-laguna=MIT
transformers=Apache-2.0
mamba=Apache-2.0
samba=MIT
zamba2=Apache-2.0
megatron-lm=NOASSERTION
qwen3=NONE
kimi-linear=MIT
minimax-01=MIT
rwkv-lm=Apache-2.0
xlstm=Apache-2.0
hymba=NONE
gpt-oss=Apache-2.0
flash-linear-attention=MIT
laguna-s=openmdw-1.1
laguna-xs=openmdw-1.1
nemotron-nano=other-nvidia-open-model
qwen3-next=apache-2.0
kimi-linear-model=mit
gpt-oss-20b=apache-2.0
olmo=Apache-2.0
olmo-core=Apache-2.0
dolma=Apache-2.0
torchtitan=BSD-3-Clause
smollm=Apache-2.0
nanogpt=MIT
lm-evaluation-harness=MIT
vllm=Apache-2.0
sglang=Apache-2.0
flashinfer=Apache-2.0
mooncake=Apache-2.0
letta=Apache-2.0
a-mem=MIT
kv-cache-compression-index=MIT
agent-memory-index=MIT
amdsense=NONE
wsl-rocm=NONE
rocm=MIT
'

declared_license() {
    local name="$1" line
    line="$(echo "$LICENSES" | grep -m1 "^${name}=")" || true
    if [ -n "$line" ]; then echo "${line#*=}"; else echo "UNRECORDED"; fi
}

clone_noninteractive() {
    # A gated repo makes git ask for credentials. Under a credential manager with no
    # console, that call blocks FOREVER rather than failing -- one gated model stalled
    # this whole fetch for 20 minutes with no error and no output.
    #
    # GIT_TERMINAL_PROMPT=0 plus an emptied credential helper turns "needs auth" into an
    # immediate, visible failure. Gated sources belong in the manifest only if someone
    # has authenticated deliberately; the library must reproduce for a stranger with no
    # HuggingFace account.
    GIT_TERMINAL_PROMPT=0 GIT_LFS_SKIP_SMUDGE=1 GCM_INTERACTIVE=never \
        git -c credential.helper= -c core.askPass= \
        clone --quiet --depth 1 --single-branch "$@" 2>/dev/null
}

neutralize_agent_instructions() {
    # Upstream repos ship their own CLAUDE.md / AGENTS.md / .cursorrules. Cloned into
    # this working tree, those are loaded as INSTRUCTIONS by any coding agent working
    # in the repo -- 40 third-party sources become 40 injection surfaces, and
    # huggingface/transformers really does ship a CLAUDE.md.
    #
    # We read this code; we do not take orders from it. Rename rather than delete: the
    # content stays readable as reference material, it just stops being auto-loaded.
    local dest="$1" found
    for pattern in 'CLAUDE.md' 'AGENTS.md' '.cursorrules' '.windsurfrules' 'GEMINI.md' 'copilot-instructions.md'; do
        while IFS= read -r found; do
            [ -n "$found" ] || continue
            mv "$found" "$found.upstream-not-instructions" 2>/dev/null &&
                echo "    neutralized $(basename "$found") in ${dest##*/}"
        done < <(find "$dest" -name "$pattern" -not -path '*/.git/*' 2>/dev/null)
    done
}

fetch_one() {
    local name="$1" category="$2" url="$3" ref="$4"
    local dest="$REF_DIR/$category/$name"

    if [ -d "$dest/.git" ]; then
        echo "  = $category/$name (present)"
        return 0
    fi

    mkdir -p "$REF_DIR/$category"
    echo "  + $category/$name <- $url ($ref)"

    # Shallow, single-branch: we read this code, we do not develop against it.
    if clone_noninteractive --branch "$ref" "$url" "$dest"; then
        neutralize_agent_instructions "$dest"
        return 0
    fi
    # Some HF repos use a default branch other than the manifest guess.
    if clone_noninteractive "$url" "$dest"; then
        echo "    (fell back to default branch)"
        neutralize_agent_instructions "$dest"
        return 0
    fi
    echo "    !! FAILED: $url" >&2
    rm -rf "$dest"
    return 1
}

write_provenance() {
    {
        echo "# PROVENANCE — everything fetched, with the revision it was fetched at"
        echo
        echo "A **register**: rows are appended and updated, never deleted. Generated by"
        echo '`scripts/fetch_reference.sh --provenance` from what is actually on disk, so a row'
        echo "here means the clone existed at that revision — not that someone intended to fetch it."
        echo
        echo "Clones are gitignored. This file plus the script reproduces the library."
        echo "Upstream LICENSE files stay intact inside each clone."
        echo
        echo "**License** is the SPDX id the GitHub/HuggingFace API reported on 2026-07-26."
        echo "\`NONE\` means no license was detected — default copyright applies, so those are"
        echo "**read-only: never vendored, never redistributed.** \`NOASSERTION\` means a license"
        echo "file exists that GitHub could not classify; read it before relying on it."
        echo "**On disk** confirms a LICENSE/COPYING file survived into the clone (hard rule 4:"
        echo "upstream LICENSEs stay intact). A \`NONE\` license with no file on disk is expected;"
        echo "a permissive license with **no** file on disk is worth a look — though for"
        echo "HuggingFace repos it is usually just convention: the license lives in the model"
        echo "card's YAML frontmatter (\`license:\`) rather than a LICENSE file. Verified for"
        echo "\`nemotron-nano\` (nvidia-open-model-license) and \`kimi-linear-model\` (mit)."
        echo
        echo "| Name | Category | URL | Revision | License | On disk | Fetched | Purpose |"
        echo "|---|---|---|---|---|---|---|---|"
    } > "$PROVENANCE"

    echo "$MANIFEST" | while IFS='|' read -r name category url ref purpose; do
        [ -z "${name:-}" ] && continue
        local dest="$REF_DIR/$category/$name"
        [ -d "$dest/.git" ] || continue
        local sha spdx on_disk fetched
        sha="$(git -C "$dest" rev-parse --short=12 HEAD 2>/dev/null || echo '?')"
        fetched="$(date -u +%Y-%m-%d)"
        spdx="$(declared_license "$name")"
        if find "$dest" -maxdepth 1 \( -iname 'LICENSE*' -o -iname 'COPYING*' \) 2>/dev/null | grep -q .; then
            on_disk="yes"
        else
            on_disk="no"
        fi
        echo "| \`$name\` | $category | $url | \`$sha\` | $spdx | $on_disk | $fetched | $purpose |" >> "$PROVENANCE"
    done

    echo "  wrote $PROVENANCE"
}

main() {
    local filter="${1:-}"

    if [ "$filter" = "--provenance" ]; then
        write_provenance
        return 0
    fi

    mkdir -p "$REF_DIR"
    local failed=0
    echo "$MANIFEST" | while IFS='|' read -r name category url ref purpose; do
        [ -z "${name:-}" ] && continue
        if [ -n "$filter" ] && [ "$filter" != "$category" ]; then continue; fi
        fetch_one "$name" "$category" "$url" "$ref" || failed=$((failed + 1))
    done

    write_provenance

    # A truncated MANIFEST once made this script print "done." after fetching 21 of 40
    # sources with exit code 0. Success that is not checked is not success: assert that
    # every manifest entry is on disk, and fail loudly when it is not.
    local expected actual missing
    expected="$(echo "$MANIFEST" | grep -c '|')"
    actual="$(grep -c '^| `' "$PROVENANCE")"
    echo "manifest: $expected   on disk: $actual"

    if [ "$expected" -ne "$actual" ]; then
        echo "INCOMPLETE -- these manifest entries did not land:" >&2
        echo "$MANIFEST" | while IFS='|' read -r name category _url _ref _purpose; do
            [ -z "${name:-}" ] && continue
            [ -d "$REF_DIR/$category/$name/.git" ] || echo "  MISSING $category/$name" >&2
        done
        echo "re-run to retry; a gated or renamed source will fail fast rather than hang." >&2
        return 1
    fi

    # Nothing upstream should be able to issue instructions to an agent working here.
    missing="$(find "$REF_DIR" \( -name 'CLAUDE.md' -o -name 'AGENTS.md' -o -name '.cursorrules' \) \
        -not -path '*/.git/*' 2>/dev/null | wc -l)"
    if [ "$missing" -ne 0 ]; then
        echo "WARNING: $missing un-neutralized upstream agent-instruction files remain" >&2
        return 1
    fi

    echo "done. $actual/$expected sources, no upstream agent-instruction files."
}

main "$@"
