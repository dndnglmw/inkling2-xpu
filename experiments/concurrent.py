"""Aggregate vs per-stream decode speed with N concurrent streamed requests (different prompts)."""
import json, sys, threading, time, urllib.request

PROMPTS = ["Write a long story about a dragon who learns to code. ", "Explain how a transformer language model works, step by step. ",
           "Describe the history of the Roman empire in detail. ", "Write a Python program that implements a binary search tree. ",
           "What are the main causes of the French Revolution? Explain. ", "Tell me a story about a lighthouse keeper. ",
           "Explain the theory of general relativity to a student. ", "List and explain ten tips for learning a new language. "]


def stream(prompt, out, i, n=160):
    req = urllib.request.Request("http://localhost:8001/v1/completions", json.dumps(
        {"model": "inkling2-mini", "prompt": prompt, "max_tokens": n, "temperature": 0, "ignore_eos": True, "stream": True}).encode(),
        {"Content-Type": "application/json"})
    times = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                times.append(time.time())
    out[i] = times


def run(nconc):
    out = [None] * nconc
    ths = [threading.Thread(target=stream, args=(PROMPTS[i], out, i)) for i in range(nconc)]
    [t.start() for t in ths]; [t.join() for t in ths]
    # steady window: from when the last stream produced its 20th token to when the first stream finished
    start = max(o[20] for o in out)
    end = min(o[-1] for o in out)
    toks = sum(sum(1 for x in o if start <= x <= end) for o in out)
    per = [sum(1 for x in o if start <= x <= end) / (end - start) for o in out]
    return toks / (end - start), per


stream("hi", [None], 0, 8) if False else None
for n in (1, 2, 4, 8):
    agg, per = run(n)
    print(f"{n} concurrent: aggregate {agg:6.1f} tok/s | per stream {sum(per) / n:5.1f} tok/s (min {min(per):.1f}, max {max(per):.1f})", flush=True)
