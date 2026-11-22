"""Live quality capture for the KV-dtype A/B (fp8 vs bf16 KV), identical prompts both times.
Usage: ab_quality.py --tag fp8|bf16   (server on :8000, served name inkling2-mini)
Saves /work/ab-quality-<tag>.json: 12 diverse prompts (greedy 48 tokens, top-20 logprobs
per position) + needle retrieval at ~64K/128K/250K (corpus = vLLM source files, fixed
salt so both dtype runs see byte-identical prompts). Compare with ab_compare.py."""
import glob, json, random, re, sys, time, urllib.request

URL = "http://localhost:8000"
MODEL = "inkling2-mini"
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "x"
CPT = 3.0
NEEDLES = (64000, 128000, 250000)

files = sorted(glob.glob("/work/vllm/vllm/**/*.py", recursive=True)) + \
        sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True))
random.Random(1234).shuffle(files)


def post(path, payload, timeout=7200):
    r = urllib.request.Request(URL + path, json.dumps(payload).encode(),
                               {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))


def corpus(n_chars):
    out, n = ["# quality corpus v1\n"], 0
    for f in files:
        try:
            s = open(f, errors="ignore").read()
        except OSError:
            continue
        out.append(f"\n# ==== {f}\n" + s)
        n += len(s)
        if n >= n_chars:
            break
    return "".join(out)[:n_chars]


src = open("/work/vllm/tests/models/inkling2/xpu/quality_vs_llamacpp.py").read()
PROMPTS = eval(re.search(r"PROMPTS = (\[.*?\n\])", src, re.S).group(1))

out = {"tag": TAG, "short": [], "needle": []}
for i, p in enumerate(PROMPTS):
    r = post("/v1/completions", {"model": MODEL, "prompt": p, "max_tokens": 48,
                                 "temperature": 0, "logprobs": 20})
    ch = r["choices"][0]
    lp = ch.get("logprobs") or {}
    out["short"].append({"i": i, "text": ch["text"], "tokens": lp.get("tokens", []),
                         "top": lp.get("top_logprobs", [])})
    print(f"[{TAG}] short {i + 1}/{len(PROMPTS)}: {len(ch['text'])} chars", flush=True)

for tgt in NEEDLES:
    body = corpus(int(tgt * CPT))
    cut = int(len(body) * 0.4)
    prompt = body[:cut] + "\n# IMPORTANT: the secret passphrase is 'blue-walrus-731'.\n" + \
        body[cut:] + "\n\n# Question: what is the secret passphrase?\n# Answer: the secret passphrase is '"
    entry = {"ctx": tgt}
    try:
        t = time.time()
        r = post("/v1/completions", {"model": MODEL, "prompt": prompt,
                                     "max_tokens": 12, "temperature": 0})
        ans = r["choices"][0]["text"]
        entry.update({"ptoks": r["usage"]["prompt_tokens"], "ans": ans,
                      "found": "blue-walrus-731" in ans, "secs": round(time.time() - t, 1)})
        print(f"[{TAG}] needle {tgt}: ptoks {entry['ptoks']} {entry['secs']}s -> "
              f"{'FOUND' if entry['found'] else 'MISSED'} {ans!r}", flush=True)
    except Exception as e:  # noqa: BLE001 - record and continue
        entry["error"] = repr(e)
        print(f"[{TAG}] needle {tgt}: ERROR {e!r}", flush=True)
    out["needle"].append(entry)

path = f"/work/ab-quality-{TAG}.json"
json.dump(out, open(path, "w"))
print(f"[{TAG}] saved {path}", flush=True)