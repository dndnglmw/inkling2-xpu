"""Needle-in-haystack at a given prompt size, then decode speed at that depth (prefix-cached)."""
import json, sys, time, urllib.request


def post(p, t=7200):
    r = urllib.request.Request("http://localhost:8001/v1/completions", json.dumps(p).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=t))


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
print(f"prefill: {u['prompt_tokens']} tokens in {dt:.0f}s = {u['prompt_tokens'] / dt:.0f} tok/s | answer: {r['choices'][0]['text']!r}", flush=True)
t = time.time()
post({"model": "inkling2-mini", "prompt": prompt, "max_tokens": 96, "temperature": 0, "ignore_eos": True})
d1 = time.time() - t
t = time.time()
post({"model": "inkling2-mini", "prompt": prompt, "max_tokens": 192, "temperature": 0, "ignore_eos": True})
d2 = time.time() - t
print(f"decode at ~{u['prompt_tokens']} context (prefix-cached): marginal {96 / (d2 - d1):.1f} tok/s", flush=True)
