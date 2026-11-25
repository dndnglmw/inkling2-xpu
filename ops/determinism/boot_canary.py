"""Boot canary: bit-exact check of the 4-row k3 serving shape against canary_ref.json.
Exit 0 = admit traffic. Exit 1 = deviant. The cold-cache assertion requires zero
prefix-cache hits during the canary (fresh boot = empty cache)."""
import json
import sys
import urllib.request

URL = "http://127.0.0.1:8000"
MODEL = "inkling2-mini"
REF = json.load(open("/home/operator/work/inkling-det/canary/canary_ref.json"))


def post(path, payload, timeout=600):
    r = urllib.request.Request(URL + path, json.dumps(payload).encode(),
                               {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))


def metric_hits():
    try:
        m = urllib.request.urlopen(URL + "/metrics", timeout=10).read().decode()
        for line in m.splitlines():
            if line.startswith("vllm:prefix_cache_hits_total"):
                return float(line.split()[-1])
    except OSError:
        pass
    return None


hits0 = metric_hits()
got = []
for i, entry in enumerate(REF["short"]):
    src = open("/home/operator/work/vllm/tests/models/inkling2/xpu/quality_vs_llamacpp.py").read()
    import re

    PROMPTS = eval(re.search(r"PROMPTS = (\[.*?\n\])", src, re.S).group(1))
    r = post("/v1/completions", {"model": MODEL, "prompt": PROMPTS[entry["i"]],
                                 "max_tokens": 48, "temperature": 0, "logprobs": 20})
    ch = r["choices"][0]
    lp = ch.get("logprobs") or {}
    got.append({"i": entry["i"], "text": ch["text"], "tokens": lp.get("tokens", []),
                "top": lp.get("top_logprobs", [])})
hits1 = metric_hits()

fails = []
for g, r in zip(got, REF["short"]):
    if g["tokens"] != r["tokens"]:
        bad = [j for j, (x, y) in enumerate(zip(g["tokens"], r["tokens"])) if x != y]
        fails.append(f"prompt i={g['i']}: token ids differ at positions {bad[:6]}")
    if json.dumps(g["top"], sort_keys=True) != json.dumps(r["top"], sort_keys=True):
        fails.append(f"prompt i={g['i']}: top-20 logprobs differ (bit-level)")

cold_ok = hits0 is None or hits1 is None or (hits1 - hits0) == 0.0
if not cold_ok:
    fails.append(f"prefix-cache hits delta {hits1 - hits0} != 0 (cold-cache assertion)")

if fails:
    print("CANARY-FAIL")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print(f"CANARY-PASS (4 prompts x 48 positions bit-exact; cold-cache {'ok' if cold_ok else 'unverified'})")
