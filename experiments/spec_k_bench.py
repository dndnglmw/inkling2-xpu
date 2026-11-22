"""True tok/s + per-position acceptance per prompt class for the running speculative server (any depth)."""
import json, os, re, sys, time, urllib.request
BASE = f"http://localhost:{os.environ.get('PORT', '8000')}"

NPOS = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def metrics():
    t = urllib.request.urlopen(BASE + "/metrics").read().decode()
    acc, drafts = {}, 0.0
    for l in t.splitlines():
        if l.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total"):
            acc[int(re.search(r'position="(\d+)"', l).group(1))] = float(l.split()[-1])
        elif l.startswith("vllm:spec_decode_num_drafts_total"):
            drafts = float(l.split()[-1])
    return acc, drafts


def gen(prompt, n=384):
    req = urllib.request.Request(BASE + "/v1/completions", json.dumps(
        {"model": "inkling2-mini", "prompt": prompt, "max_tokens": n, "temperature": 0, "ignore_eos": True}).encode(),
        {"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    return r["usage"]["completion_tokens"], time.time() - t


prompts = {"prose": ["Write a long story about a dragon who learns to code. ", "Tell me a story about a lighthouse keeper. "],
           "facts": ["Explain in detail how photosynthesis works, step by step.\n\n", "Describe the history of the Roman empire in detail. "],
           "chat": ["Q: What are the main differences between TCP and UDP?\nA:", "Q: How do I set up a Python virtual environment?\nA:"],
           "code": ["Write a Python function that parses a CSV file and returns the rows as dictionaries, with error handling.\n\n```python\n",
                    "Write a Python program that implements a binary search tree with insert, delete and traversal.\n\n```python\n"]}
gen("hi", 8)
means = []
print(f"{'class':6s} " + " ".join(f"p{i}".rjust(5) for i in range(NPOS)) + " | tokens/step | tok/s (true) | step ms")
for name, ps in prompts.items():
    a0, d0 = metrics()
    toks = secs = 0.0
    for p in ps:
        n, dt = gen(p)
        toks += n; secs += dt
    a1, d1 = metrics()
    nd = max(1.0, d1 - d0)
    pr = [(a1.get(i, 0) - a0.get(i, 0)) / nd for i in range(NPOS)]
    tps = toks / secs
    means.append(tps)
    tps_step = 1 + sum(pr)
    print(f"{name:6s} " + " ".join(f"{x:5.2f}" for x in pr) + f" | {tps_step:11.2f} | {tps:13.1f} | {1000 * tps_step / tps:6.1f}", flush=True)
print(f"mean over the 4 classes: {sum(means) / len(means):.1f} tok/s")
