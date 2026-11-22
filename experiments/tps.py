import json, time, urllib.request


def post(p):
    r = urllib.request.Request("http://localhost:8001/v1/completions", json.dumps(p).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))


post({"model": "inkling2-mini", "prompt": "hi", "max_tokens": 4})
r = post({"model": "inkling2-mini", "prompt": "The capital of France is", "max_tokens": 24, "temperature": 0})
print(repr(r["choices"][0]["text"]))
t = time.time()
r = post({"model": "inkling2-mini", "prompt": "Write a long story about a dragon. ", "max_tokens": 128, "temperature": 0, "ignore_eos": True})
dt = time.time() - t
print("128 tok: %.2fs -> %.1f tok/s" % (dt, 128 / dt))
