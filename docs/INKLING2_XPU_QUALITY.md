# Inkling 2 Mini on Intel XPU — quality evidence (consolidated)

Date: 2026-11-22. Complements `INKLING2_XPU_HANDOFF.md` and
`INKLING2_XPU_GITHUB_COMPARISON.md`. There was **no single quality document**
before this file — the evidence existed but was scattered across scripts, JSON
dumps and logs. Everything below was re-computed or re-read from the artifacts
today; nothing is quoted from memory.

---

## 1. TL;DR

| Check | Result | Source |
|---|---|---|
| Step-0 distribution vs llama.cpp (3-pass loop), BF16 ref | **corr 0.9995** over top-128, identical top-8 | session-1 oracle (`compare_step0.py`) |
| 16-token greedy vs llama.cpp, BF16 ref | **16/16 top-1**, 126/128 top-8 | session-1 (`compare_llamacpp.py`) |
| Optimisation A/Bs (fp8-dense on/off, fold, prefill-opt, MoE-Task-A, fusion) | **12/12 top-1 agreement, mean KL ≤ 0.0025, max KL ≤ 0.014** between every pair (computed today from dumps) | `agents/decode/dump_*.json` + `compare_dumps.py` |
| Needle-in-haystack retrieval on the final 4-bit TP2 stack | **FOUND at 64K / 128K / 200K / 246K / 494K / 940K tokens** | `agents/needles_fold.txt`, `agents/final_pass.txt` |
| Kernel test gates (every delegated change) | ALL PASS: attn rel 1.5–2.1e-3 (gate 2e-2), grouped MoE rel 1.7–2.4e-4 (gate 5e-3), decode MoE rel 0.0–2.2e-4 (gate 2e-3) | the three `REPORT.md`s |
| Prompt-0 greedy fork (see §4) | one 24-token greedy diverges at token 2 after a Nov-21 morning change — the new phrasing **matches the llama.cpp oracle's style**, the old did not | computed today |
| What is *missing* | no GSM8K/MMMU-style eval on XPU; no saved run of `quality_vs_llamacpp.py` against the live server; no long-form/human eval | §6 |

## 2. Quality harnesses that exist (and how to run them)

| Tool | What it measures |
|---|---|
| `vllm/tests/models/inkling2/xpu/quality_vs_llamacpp.py` | Live A/B: vLLM XPU server (`--vllm`, default :8001) vs llama.cpp 3-pass oracle (`--llama`, default :8099) on 12 fixed prompts; top-k agreement + KL over shared mass. **This is the direct oracle check — no saved output of it exists yet.** |
| `agents/decode/quality_dump.py` / `quality_dump_http.py` | Offline/server dump of top-40 next-token logprobs for the same 12 prompts (+24-token greedy for the first 3) → JSON. |
| `agents/decode/compare_dumps.py <a> <b>` | Two dumps → top-1 agreement, mean/max KL(old‖new) over shared top-40 mass, first-divergent greedy token. |
| `agents/needle_corpus.py` (+ `agents/needles_fold.txt`, `final_pass.txt`) | Needle-in-a-haystack at 64K–1M; the answer string must appear in the completion. |
| `vllm/vllm/models/inkling2/xpu/kv_sim.py` + `agents/kv_quality.py` | **Simulated** KV-cache quantisation for quality experiments only (`INKLING_KV_SIM=off|fp8|rtn<bits>|tq<bits>|k8v4...`; rtn = per-vector absmax, tq = Hadamard + Lloyd-Max). No saved results yet. |
| `tests/models/inkling2/xpu/test_*.py` | Per-kernel numeric gates vs torch references (see §5). |

## 3. Optimisation A/B lineage (checksum-resolved)

All dumps are the same 12 prompts / top-40. Checksum groups:

| Group (md5) | Files | Meaning |
|---|---|---|
| `dd2b…` | `dump_old.json` (12:39) | pre-change baseline of Nov 21 morning |
| `e0e2…` | `dump_new.json` (12:39) | after that morning's change (unnamed in artifacts) |
| `c1c4…` | `dump_final.json` (13:04) | reference for the later A/Bs |
| `c21d…` | `dump_final_nodense`, `dump_fold`, `dump_prefillopt` | **math-identical**: fp8-dense OFF ≡ fold ≡ prefill-opt |
| `4bdc…` | `dump_afterA`, `dump_final_fusion`, `dump_fusion` | **math-identical**: post-Task-A ≡ fusion configs |

Pairwise results (computed 2026-11-22 with `compare_dumps.py`):

| Pair | top-1 agree | mean KL | max KL | greedy divergence |
|---|---|---|---|---|
| old → new | 12/12 | 0.00045 | 0.0025 | prompt 0 at token 2; prompts 1–2 identical 24/24 |
| old → final | 12/12 | 0.00239 | 0.0138 | same prompt-0 fork |
| final → nodense/fold/prefillopt | 12/12 | 0.00248 | 0.0139 | same fork |
| final → fusion/afterA | 12/12 | 0.00248 | 0.0139 | same fork |

Reading: **every accepted optimisation is quality-neutral** at top-40/KL
granularity — the fp8-dense toggle, fold, prefill opts, MoE bmm path and kernel
fusions all land in the same ≈0.0025-KL band (comparable to the reference's own
FlashAttention-vs-PyTorch spread upstream reported). The `c21d…` and `4bdc…`
groups being internally bit-identical is strong evidence the fold/prefill
changes are pure scheduling, not math.

## 4. The prompt-0 greedy fork (the only behavioral change on record)

Prompt: `"The capital of France is Paris.\n\nA."` (greedy, 24 tokens):

```
old  :  " Proper Noun\n\nB. Common Noun\n\nC…"
final:  " Noun clause\n\nB. Adjective clause\n\nC…"
```

- The fork appeared between `dump_old` and `dump_new` (both written 12:39 on
  Nov 21; the generating change is not named in any surviving artifact — the
  candidates in the tree from that morning are the attention far-region
  rewrite (`triton_attention.py.pre-vendorfar` backup) and the fp8-dense
  toggle).
- The **new** phrasing matches the llama.cpp oracle's continuation style
  verified in session 1 (`"A. Noun clause B. Adjective clause"`), i.e. the
  change moved *toward* the oracle, not away. Still, an exact re-check of this
  prompt against the live llama.cpp server is a 2-minute task and worth doing
  (§6).

## 5. Kernel-level numeric gates (from the delegated-task reports)

| Kernel / test | Accuracy vs reference | Gate |
|---|---|---|
| `test_triton_attention.py` (11 cases incl. descriptor far path) | rel 0.0015–0.0021 | 2e-2 |
| `test_triton_moe_grouped.py` (real checkpoint tensors) | rel 1.7e-4–2.4e-4, identical to baseline | 5e-3 |
| `test_triton_moe_q.py` decode | rel 0.00000 (NVFP4) / 0.00022 (MXFP8) | 2e-3 |
| `test_triton_moe_sg.py` | rel ≤ 3.3e-4 | 2e-3 |
| decode fp4 decode helper | exact against `_e2m1`/`_e4m3` for **all 65,536 uint16 patterns** (max diff 0) | exact |
| graph capture | replay output rel 0.0 | exact |

Session-1 end-to-end gates (BF16 reference era): `test_reference_tiny.py`
invariants pass; 1-pass trace vs HF oracle: layers 0–22 within drift, final
rel_sq 0.0009 embed / ≤ 0.05 per layer; step-0 corr 0.9985 (1-pass) / 0.9995
(3-pass).

## 6. Gaps — what quality evidence is still missing

1. **Direct server-vs-oracle number on the final stack.** Run
   `quality_vs_llamacpp.py --vllm http://localhost:8001 --llama http://localhost:8099`
   with the production config (4-bit experts, fp8 KV, DSpark idle) and save the
   output. The llama.cpp 3-pass oracle was still running on port 8099 as of
   Nov 21 evening.
2. **Benchmark-style evals.** Upstream #69028 reports GSM8K 97.0 % (200 q,
   DSpark) and needle 550/550 (4K–1M × 11 depths); we have single-depth needle
   runs only. A 200-question GSM8K at TP2 with DSpark would be the cheapest
   apples-to-apples number (the B70 cookbook's gate was greedy-vs-llama.cpp 5/5).
3. **KV-cache quant quality.** `kv_sim.py` supports fp8/rtn/tq simulations but
   no results were saved; the production stack runs `kv_cache_dtype=fp8`, so a
   sim-vs-off KL table would quantify its cost.
4. **Prompt-0 re-check** against the live oracle (§4).
5. Long-form coherence / multi-turn beyond the 12 prompts (all current evidence
   is next-token + needle + per-kernel rel-error).

## 7. One-line summary

Every accepted change since the BF16 reference is measured quality-neutral
(12/12 top-1, KL ≤ 0.0025), needles are found out to 940K tokens on the final
4-bit TP2 + fp8-KV + DSpark stack, and all kernel gates pass — but the project
has no saved eval-suite numbers (GSM8K etc.) and the direct
vLLM-vs-llama.cpp server comparison has never been captured to a file.