# SPDX-License-Identifier: Apache-2.0
"""Next-token distribution agreement: vLLM XPU server vs llama.cpp (3-pass oracle).

Usage: python quality_vs_llamacpp.py [--vllm http://localhost:8001] [--llama http://localhost:8099]
"""
import argparse, json, math, urllib.request

PROMPTS = [
    "The capital of France is Paris.\n\nA.",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "Q: What is 17 * 23?\nA: 17 * 23 =",
    "The mitochondria is the powerhouse of the",
    "In 1969, Neil Armstrong became the first person to",
    "SELECT name, COUNT(*) FROM users GROUP BY",
    "Once upon a time, in a small village by the sea, there lived a",
    "import numpy as np\nx = np.zeros((3, 3))\nprint(x.",
    "Translate to French: 'The weather is nice today.' ->",
    "The three primary colors are red, blue, and",
    "To install the package, run: pip install",
    "Photosynthesis converts light energy into chemical energy, producing glucose and",
]


def post(url, payload):
    r = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))


def vllm_top(base, prompt, k):
    r = post(base + "/v1/completions", {"model": "inkling2-mini", "prompt": prompt, "max_tokens": 1,
                                         "temperature": 0, "logprobs": k})
    return r["choices"][0]["logprobs"]["top_logprobs"][0]


def llama_top(base, prompt, k):
    r = post(base + "/completion", {"prompt": prompt, "n_predict": 1, "temperature": 0, "n_probs": k, "cache_prompt": False})
    return {e["token"]: e["logprob"] for e in r["completion_probabilities"][0]["top_logprobs"]}


ap = argparse.ArgumentParser()
ap.add_argument("--vllm", default="http://localhost:8001")
ap.add_argument("--llama", default="http://localhost:8099")
ap.add_argument("--k", type=int, default=20)
a = ap.parse_args()
agree, kls, diffs = 0, [], []
for p in PROMPTS:
    v, l = vllm_top(a.vllm, p, a.k), llama_top(a.llama, p, a.k)
    vt, lt = max(v, key=v.get), max(l, key=l.get)
    shared = [t for t in v if t in l]
    # KL(llama || vllm) over shared mass, renormalised
    pl = {t: math.exp(l[t]) for t in shared}; pv = {t: math.exp(v[t]) for t in shared}
    zl, zv = sum(pl.values()), sum(pv.values())
    kl = sum((pl[t] / zl) * math.log((pl[t] / zl) / (pv[t] / zv)) for t in shared)
    d = sum(abs(v[t] - l[t]) for t in shared[:5]) / max(1, len(shared[:5]))
    agree += vt == lt; kls.append(kl); diffs.append(d)
    print(f"{'OK ' if vt == lt else 'DIF'} top1 vllm={vt!r:12} llama={lt!r:12} KL={kl:.4f} meanAbsLogprobDiff(top5)={d:.3f} | {p[:40]!r}")
print(f"top-1 agreement {agree}/{len(PROMPTS)}  mean KL {sum(kls)/len(kls):.4f}  mean |dlogp| {sum(diffs)/len(diffs):.3f}")
