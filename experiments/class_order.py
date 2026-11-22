"""Is the code-class step slowdown tied to the class or to run order? Runs classes in the given order, twice."""
import json, os, re, sys, time, urllib.request

BASE = f"http://localhost:{os.environ.get('PORT', '8000')}"
P = {"prose": "Write a long story about a dragon who learns to code. ",
     "chat": "Q: What are the main differences between TCP and UDP?\nA:",
     "code": "Write a Python function that parses a CSV file and returns the rows as dictionaries, with error handling.\n\n```python\n",
     "code2": "Write a Python program that implements a binary search tree with insert, delete and traversal.\n\n```python\n"}


def metrics():
    t = urllib.request.urlopen(BASE + "/metrics").read().decode()
    a = d = 0.0
    for l in t.splitlines():
        if l.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            a = float(l.split()[-1])
        elif l.startswith("vllm:spec_decode_num_drafts_total"):
            d = float(l.split()[-1])
    return a, d


def gen(prompt, n=384):
    req = urllib.request.Request(BASE + "/v1/completions", json.dumps(
        {"model": "inkling2-mini", "prompt": prompt, "max_tokens": n, "temperature": 0, "ignore_eos": True}).encode(),
        {"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    return r["usage"]["completion_tokens"], time.time() - t


for rep in range(2):
    for name in sys.argv[1:]:
        a0, d0 = metrics()
        n, dt = gen(P[name])
        a1, d1 = metrics()
        tps = n / dt
        tp = 1 + (a1 - a0) / max(1, d1 - d0)
        print(f"rep{rep} {name:6s} {tps:6.1f} tok/s | {tp:.2f} tok/step | step {1000 * tp / tps:5.1f} ms", flush=True)
