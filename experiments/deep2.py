"""Needle at a given prompt size, then true decode tok/s + tokens/step + step ms at that depth (prefix-cached, spec server)."""
import json, re, sys, time, urllib.request

import os
URL = f"http://localhost:{os.environ.get('PORT', '8000')}"


def post(p, t=7200):
    r = urllib.request.Request(URL + "/v1/completions", json.dumps(p).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=t))


def metrics():
    t = urllib.request.urlopen(URL + "/metrics").read().decode()
    acc = drafts = 0.0
    for l in t.splitlines():
        if l.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            acc = float(l.split()[-1])
        elif l.startswith("vllm:spec_decode_num_drafts_total"):
            drafts = float(l.split()[-1])
    return acc, drafts


n_tok = int(sys.argv[1])
filler = "The quick brown fox jumps over the lazy dog near the river bank while the sun sets slowly. "
reps = n_tok // 20
pos = int(reps * 0.37)
body = filler * pos + "IMPORTANT: the secret passphrase is 'purple-giraffe-42'. " + filler * (reps - pos)
prompt = body + "\n\nQuestion: What is the secret passphrase?\nAnswer: The secret passphrase is '"
t = time.time()
r = post({"model": "inkling2-mini", "prompt": prompt, "max_tokens": 12, "temperature": 0})
dt = time.time() - t
u = r["usage"]
txt = r["choices"][0]["text"]
print(f"[{n_tok}] prefill: {u['prompt_tokens']} tokens in {dt:.0f}s = {u['prompt_tokens'] / dt:.0f} tok/s | answer: {txt!r} | needle {'FOUND' if 'purple-giraffe-42' in txt else 'MISSED'}", flush=True)
post({"model": "inkling2-mini", "prompt": prompt, "max_tokens": 8, "temperature": 0})  # warm the prefix cache for the full prompt
a0, d0 = metrics()
t = time.time()
r = post({"model": "inkling2-mini", "prompt": prompt, "max_tokens": 256, "temperature": 0, "ignore_eos": True})
dt = time.time() - t
a1, d1 = metrics()
n = r["usage"]["completion_tokens"]
per = (a1 - a0) / max(1, d1 - d0)
tps = n / dt
print(f"[{n_tok}] decode at ~{u['prompt_tokens']} ctx: {tps:.1f} tok/s (true, incl. cache-hit overhead) | {1 + per:.2f} tokens/step | step {1000 * (1 + per) / tps:.1f} ms", flush=True)
