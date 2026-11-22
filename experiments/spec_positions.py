"""Per-position acceptance for each prompt class at depth 7, then the oracle best depth from a step-cost model.

Step cost c(T) (ms, measured by concurrent decode at TP2 graphs): T=1 20.3, 2 23.9, 4 30.7, 8 43.5 (interpolated linearly in between);
draft cost d is taken from the measured depth-7 step (48.3 ms) minus c(8).
"""
import json, re, time, urllib.request

C = {1: 20.3, 2: 23.9, 3: 27.3, 4: 30.7, 5: 34.0, 6: 37.3, 7: 40.4, 8: 43.5}
D_DRAFT = 48.3 - 43.5


def metrics():
    t = urllib.request.urlopen("http://localhost:8001/metrics").read().decode()
    acc, drafts = {}, 0.0
    for l in t.splitlines():
        if l.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total"):
            pos = int(re.search(r'position="(\d+)"', l).group(1))
            acc[pos] = float(l.split()[-1])
        elif l.startswith("vllm:spec_decode_num_drafts_total"):
            drafts = float(l.split()[-1])
    return acc, drafts


def gen(prompt, n=384):
    req = urllib.request.Request("http://localhost:8001/v1/completions", json.dumps(
        {"model": "inkling2-mini", "prompt": prompt, "max_tokens": n, "temperature": 0, "ignore_eos": True}).encode(),
        {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))


prompts = {"prose": ["Write a long story about a dragon who learns to code. ", "Tell me a story about a lighthouse keeper. "],
           "facts": ["Explain in detail how photosynthesis works, step by step.\n\n", "Describe the history of the Roman empire in detail. "],
           "chat": ["Q: What are the main differences between TCP and UDP?\nA:", "Q: How do I set up a Python virtual environment?\nA:"],
           "code": ["Write a Python function that parses a CSV file and returns the rows as dictionaries, with error handling.\n\n```python\n",
                    "Write a Python program that implements a binary search tree with insert, delete and traversal.\n\n```python\n"]}
gen("hi", 8)
print(f"{'class':6s} " + " ".join(f"p{i}" .rjust(5) for i in range(7)) + " | tok/s by verified depth k (k=0 plain 49.3):  " + "  ".join(f"k={k}" for k in range(8)))
for name, ps in prompts.items():
    a0, d0 = metrics()
    for p in ps:
        gen(p)
    a1, d1 = metrics()
    n = max(1.0, d1 - d0)
    pr = [(a1.get(i, 0) - a0.get(i, 0)) / n for i in range(7)]  # P(position i accepted) per draft
    row = []
    for k in range(8):
        tokens = 1 + sum(pr[:k])
        step = (C[1] + 0.0) if k == 0 else C[k + 1] + D_DRAFT
        row.append(1000 * tokens / step)
    best = max(range(8), key=lambda k: row[k])
    print(f"{name:6s} " + " ".join(f"{x:5.2f}" for x in pr) + " | " + "  ".join(f"{r:5.1f}" for r in row) + f"  -> best k={best} ({row[best]:.1f}; depth7 {row[7]:.1f})", flush=True)
