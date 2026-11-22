"""Same output format as quality_dump.py, but against the running server (final state: spec depth, fp8 dense, fp8 KV)."""
import json, os, re, sys, urllib.request

out_path = sys.argv[1]
src = open("/work/vllm/tests/models/inkling2/xpu/quality_vs_llamacpp.py").read()
PROMPTS = eval(re.search(r"PROMPTS = (\[.*?\n\])", src, re.S).group(1))


def call(prompt, n, lp):
    req = urllib.request.Request(f"http://localhost:{os.environ.get('PORT', '8000')}/v1/completions", json.dumps(
        {"model": "inkling2-mini", "prompt": prompt, "max_tokens": n, "temperature": 0, "logprobs": lp,
         "return_tokens_as_token_ids": True, "ignore_eos": True}).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["logprobs"]


tid = lambda s: s.split(":")[-1]
res = {"dist": [], "greedy": []}
for p in PROMPTS:
    top = call(p, 1, 20)["top_logprobs"][0]
    res["dist"].append({tid(t): l for t, l in top.items()})
for p in PROMPTS[:3]:
    res["greedy"].append([int(tid(t)) for t in call(p, 24, 1)["tokens"]])
json.dump(res, open(out_path, "w"))
print("saved", out_path)
