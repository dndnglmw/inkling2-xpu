"""Decode step cost vs context on the spec server. Marginal timing (256 vs 96 tokens, prefix-cached) removes cache-hit overhead.
Prompts are nested prefixes of one filler text so each depth reuses the previous prefill. Also reports prefix-cache hit counters."""
import json, sys, time, urllib.request

import os
URL = f"http://localhost:{os.environ.get('PORT', '8000')}"
FILL = "The quick brown fox jumps over the lazy dog near the river bank while the sun sets slowly. "


def post(p):
    r = urllib.request.Request(URL + "/v1/completions", json.dumps(p).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=7200))


def metric(prefix):
    t = urllib.request.urlopen(URL + "/metrics").read().decode()
    return sum(float(l.split()[-1]) for l in t.splitlines() if l.startswith(prefix) and "created" not in l.split()[0])


def spec():
    return metric("vllm:spec_decode_num_accepted_tokens_total"), metric("vllm:spec_decode_num_drafts_total")


for n_tok in (int(a) for a in sys.argv[1:]):
    prompt = FILL * (n_tok // 20) + "\n\nContinue the passage above in your own words:"
    t = time.time()
    r = post({"model": "inkling2-mini", "prompt": prompt, "max_tokens": 4, "temperature": 0})
    pre = time.time() - t
    ptoks = r["usage"]["prompt_tokens"]
    h0, q0 = metric("vllm:prefix_cache_hits_total"), metric("vllm:prefix_cache_queries_total")
    a0, d0 = spec()
    res = {}
    for n in (96, 256):
        t = time.time()
        post({"model": "inkling2-mini", "prompt": prompt, "max_tokens": n, "temperature": 0, "ignore_eos": True})
        res[n] = time.time() - t
    a1, d1 = spec()
    h1, q1 = metric("vllm:prefix_cache_hits_total"), metric("vllm:prefix_cache_queries_total")
    tps = (256 - 96) / (res[256] - res[96])
    tpstep = 1 + (a1 - a0) / max(1, d1 - d0)
    print(f"ctx {ptoks:7d} | prefill/first call {pre:5.0f}s | decode {tps:5.1f} tok/s (marginal) | {tpstep:.2f} tok/step | step {1000 * tpstep / tps:6.1f} ms | "
          f"cache-hit overhead {res[96] - 96 / tps:4.1f}s | prefix hit rate in decode calls {(h1 - h0) / max(1, q1 - q0):.3f}", flush=True)
