"""Steady decode speed: 3 streamed 128-token generations, token-to-token gaps (excludes prefill); prints text hash."""
import hashlib, json, sys, time, urllib.request

prompt = sys.argv[1] if len(sys.argv) > 1 else "Write a long story about a dragon who learns to code. "


def stream(prompt, n=128):
    req = urllib.request.Request("http://localhost:8001/v1/completions", json.dumps(
        {"model": "inkling2-mini", "prompt": prompt, "max_tokens": n, "temperature": 0, "ignore_eos": True, "stream": True}).encode(),
        {"Content-Type": "application/json"})
    times, text = [], ""
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                times.append(time.time())
                text += json.loads(line[6:])["choices"][0]["text"]
    return times, text


stream("hi", 8)  # warm up
res = []
for i in range(3):
    times, text = stream(prompt)
    gaps = [b - a for a, b in zip(times, times[1:])][8:]
    res.append(len(gaps) / sum(gaps))
print("decode tok/s: " + " / ".join(f"{r:.1f}" for r in res) + f" | mean {sum(res) / 3:.1f} | text md5 {hashlib.md5(text.encode()).hexdigest()[:8]}")
