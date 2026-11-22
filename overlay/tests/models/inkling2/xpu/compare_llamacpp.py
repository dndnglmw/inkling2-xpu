# SPDX-License-Identifier: Apache-2.0
"""Compare the Inkling2 XPU reference against llama.cpp (ground truth).

Protocol:
1. llama-server (Q8_0 GGUF) tokenizes a fixed prompt (/tokenize) and greedily
   generates N tokens with top-k alternatives (/completion, n_probs).
2. This script runs the same token ids through the pure-torch reference
   (BF16 dequantized checkpoint) greedily, recording top-k per step.
3. Reports per-step top-1 agreement and top-k overlap.
"""

import json
import os
import sys
import urllib.request

import torch

sys.path.insert(0, os.path.expanduser("~/work/vllm"))
from vllm.models.inkling2.xpu.reference import Inkling2Reference, load_checkpoint  # noqa: E402

PORT = 8099
PROMPT = "The capital of France is"
N_PREDICT = 16
TOP_K = 8


def api(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main() -> None:
    tok = api("/tokenize", {"content": PROMPT})
    ids = tok["tokens"]
    print(f"llama.cpp tokenization ({len(ids)} ids): {ids}")

    comp = api(
        "/completion",
        {
            "prompt": PROMPT,
            "n_predict": N_PREDICT,
            "temperature": 0.0,
            "n_probs": TOP_K,
            "cache_prompt": False,
        },
    )
    lc_tokens = comp["tokens"]
    lc_probs = comp["completion_probabilities"]
    # each entry: {"id", "token", "logprob", "top_logprobs": [{"id", "token", "logprob"}, ...]}
    lc_top = [[e["id"] for e in step["top_logprobs"]][:TOP_K] for step in lc_probs]
    lc_gap = [
        step["top_logprobs"][0]["logprob"]
        - (step["top_logprobs"][1]["logprob"] if len(step["top_logprobs"]) > 1 else float("-inf"))
        for step in lc_probs
    ]
    print("llama.cpp greedy:", "".join(lc_tokens))
    print("llama.cpp top-1s :", lc_top[0][:1], "...")

    print("loading reference checkpoint (49 GiB, bf16)...")
    cfg, weights = load_checkpoint(os.path.expanduser("~/work/models/inkling-bf16"))
    model = Inkling2Reference(cfg, weights)

    from tokenizers import Tokenizer

    tokz = Tokenizer.from_file(os.path.expanduser("~/work/models/inkling-hf/tokenizer.json"))
    token_strs: dict[int, str] = {}

    def piece(t: int) -> str:
        return token_strs.setdefault(t, tokz.decode([t]))

    generated: list[int] = []
    ref_top: list[list[int]] = []
    logits = model.prefill(torch.tensor(ids))
    for step in range(N_PREDICT):
        top = torch.topk(logits[-1], TOP_K)
        ref_top.append(top.indices.tolist())
        nxt = top.indices[0].item()
        generated.append(nxt)
        if step + 1 < N_PREDICT:
            logits = model.decode(nxt)
    print("reference greedy:", repr("".join(piece(t) for t in generated)))
    print("reference top-1s :", [piece(t[0]) for t in ref_top])

    agree = 0
    overlap = 0
    for i, (step, lc, rc) in enumerate(zip(lc_probs, lc_top, ref_top)):
        if lc[0] == rc[0]:
            agree += 1
        else:
            lc_shown = [(e["token"], round(e["logprob"], 3)) for e in step["top_logprobs"][:3]]
            ref_shown = [piece(t) for t in rc[:3]]
            print(f"  step {i}: lc={lc_shown} ref={ref_shown} (gap {lc_gap[i]:.3f})")
        overlap += len(set(lc) & set(rc))
    print(f"top-1 agreement: {agree}/{N_PREDICT}")
    print(f"top-{TOP_K} overlap: {overlap}/{N_PREDICT * TOP_K}")


if __name__ == "__main__":
    main()