"""Speculative decode: steady tok/s (streamed) and acceptance per prompt type."""
import json, re, sys, time, urllib.request


def metric(name):
    t = urllib.request.urlopen("http://localhost:8001/metrics").read().decode()
    return sum(float(l.split()[-1]) for l in t.splitlines() if l.startswith(name) and not l.startswith("#") and "created" not in l.split()[0])


def stream(prompt, n=200):
    req = urllib.request.Request("http://localhost:8001/v1/completions", json.dumps(
        {"model": "inkling2-mini", "prompt": prompt, "max_tokens": n, "temperature": 0, "ignore_eos": True, "stream": True}).encode(),
        {"Content-Type": "application/json"})
    times = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                times.append(time.time())
    return times


stream("hi", 8)
prompts = {"prose": "Write a long story about a dragon who learns to code. ",
           "code": "Write a Python function that parses a CSV file and returns the rows as dictionaries, with error handling.\n\n```python\n",
           "facts": "Explain in detail how photosynthesis works, step by step.\n\n",
           "chat": "Q: What are the main differences between TCP and UDP?\nA:"}
tot = []
for name, p in prompts.items():
    a0, d0 = metric("vllm:spec_decode_num_accepted_tokens_total"), metric("vllm:spec_decode_num_drafts_total")
    t0 = time.time()
    req = urllib.request.Request("http://localhost:8001/v1/completions", json.dumps(
        {"model": "inkling2-mini", "prompt": p, "max_tokens": 256, "temperature": 0, "ignore_eos": True}).encode(),
        {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    dt = time.time() - t0
    a1, d1 = metric("vllm:spec_decode_num_accepted_tokens_total"), metric("vllm:spec_decode_num_drafts_total")
    tps = r["usage"]["completion_tokens"] / dt  # true tokens (prompt is short, so prefill is negligible)
    tot.append(tps)
    per = (a1 - a0) / max(1, d1 - d0)
    print(f"{name:6s}: {tps:6.1f} tok/s steady | accepted/draft {per:.2f} -> {1 + per:.2f} tokens/step", flush=True)
print(f"mean over the 4 prompt types: {sum(tot) / len(tot):.1f} tok/s")
