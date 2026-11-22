# SPDX-License-Identifier: Apache-2.0
"""Step-0 distribution comparison: reference vs llama.cpp for a given pass count."""

import argparse
import json
import os
import sys
import urllib.request

import torch

sys.path.insert(0, os.path.expanduser("~/work/vllm"))
from vllm.models.inkling2.xpu.reference import Inkling2Reference, load_checkpoint  # noqa: E402


def api(port: int, payload: dict) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--topk", type=int, default=128)
    args = ap.parse_args()

    comp = api(
        args.port,
        {
            "prompt": args.prompt,
            "n_predict": 1,
            "temperature": 0.0,
            "n_probs": args.topk,
            "cache_prompt": False,
        },
    )
    step = comp["completion_probabilities"][0]
    lc = step["top_logprobs"]
    ids = [e["id"] for e in lc]
    print("llama.cpp top8:", [(e["token"], round(e["logprob"], 3)) for e in lc[:8]])

    tokz_req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/tokenize",
        data=json.dumps({"content": args.prompt}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(tokz_req, timeout=60) as r:
        prompt_ids = json.loads(r.read())["tokens"]
    print("prompt ids:", prompt_ids)

    cfg, weights = load_checkpoint(os.path.expanduser("~/work/models/inkling-bf16"))
    cfg.num_passes = args.passes
    model = Inkling2Reference(cfg, weights)
    logits = model.prefill(torch.tensor(prompt_ids))[-1]
    lp = logits - torch.logsumexp(logits, 0)

    from tokenizers import Tokenizer

    tokz = Tokenizer.from_file(os.path.expanduser("~/work/models/inkling-hf/tokenizer.json"))
    top = torch.topk(lp, 8)
    print("reference top8:", [(tokz.decode([t]), round(float(v), 3)) for v, t in zip(top.values, top.indices)])

    # rank of llama.cpp's top-1 in the reference, and correlation over the top-128
    lc1 = ids[0]
    rank = int((lp > lp[lc1]).sum())
    print(f"lc top-1 {lc1} ({step['token']!r}) rank in ref: {rank}, ref lp {float(lp[lc1]):.3f} vs lc {lc[0]['logprob']:.3f}")
    a = torch.tensor([lp[i].item() for i in ids])
    b = torch.tensor([e["logprob"] for e in lc])
    corr = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    print(f"corr over top-{args.topk}: {corr:.4f}; mean diff {float((a - b).mean()):.3f} +- {float((a - b).std()):.3f}")
    ref1 = int(top.indices[0])
    print(f"ref top-1 {ref1} ({tokz.decode([ref1])!r}) in lc top-{args.topk}:",
          any(i == ref1 for i in ids))


if __name__ == "__main__":
    main()
