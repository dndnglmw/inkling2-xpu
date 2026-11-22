"""Decode speed at depth, measured as token-to-token gaps in a streamed generation (excludes prefill / cache lookup)."""
import json, sys, time, urllib.request

n_tok = int(sys.argv[1])
filler = "The quick brown fox jumps over the lazy dog near the river bank while the sun sets slowly. "
reps = n_tok // 20
pos = int(reps * 0.37)
body = filler * pos + "IMPORTANT: the secret passphrase is 'purple-giraffe-42'. " + filler * (reps - pos)
prompt = body + "\n\nQuestion: What is the secret passphrase?\nAnswer: The secret passphrase is '"


def stream(max_tokens):
    req = urllib.request.Request("http://localhost:8001/v1/completions", json.dumps(
        {"model": "inkling2-mini", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True, "stream": True}).encode(),
        {"Content-Type": "application/json"})
    times = []
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=7200) as r:
        for line in r:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                times.append(time.time())
    return t0, times


for attempt in range(2):  # 1st call warms the prefix cache / any one-off recompute; 2nd is the measurement
    t0, times = stream(96)
    gaps = [b - a for a, b in zip(times, times[1:])][8:]  # skip the first tokens
    first = times[0] - t0
    print(f"attempt {attempt}: first token after {first:.2f}s ({len(times)} chunks) | steady decode: "
          f"{len(gaps) / sum(gaps):.1f} tok/s (median gap {sorted(gaps)[len(gaps) // 2] * 1e3:.1f} ms, max {max(gaps) * 1e3:.0f} ms)", flush=True)
