"""Real acceptance + step cost on diverse long text. Context = source-file corpus (cold prefill once, then prefix-cached);
tails: code continuation, prose summary, verbatim quote. Marginal tok/s (256 vs 96 tokens) + tokens/step from the spec metrics.
Usage: deep4.py CONTEXT_TOKENS [...]   (PORT env, default 8000)"""
import glob, json, os, random, sys, time, urllib.request

URL = f"http://localhost:{os.environ.get('PORT', '8000')}"
files = sorted(glob.glob("/work/vllm/vllm/**/*.py", recursive=True)) + sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True))
random.Random(1234).shuffle(files)
CPT = float(os.environ.get("CPT", "3.0"))
SALT = os.environ.get("SALT") or random.Random(int(time.time())).random()  # fixed SALT => identical prompts across builds


def post(p):
    r = urllib.request.Request(URL + "/v1/completions", json.dumps(p).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=7200))


def metric(prefix):
    t = urllib.request.urlopen(URL + "/metrics").read().decode()
    return sum(float(l.split()[-1]) for l in t.splitlines() if l.startswith(prefix) and "created" not in l.split()[0])


def corpus(n_chars):
    out, n = [f"# run {SALT} {n_chars}\n"], 0
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


def first_file_head():
    f = files[0]
    return f, open(f, errors="ignore").read()


TAILS = {
    "code-continue": lambda body: body + "\n# ==== /work/new_module.py\nimport os\nimport sys\nimport json\n\n\nclass ConfigLoader:\n    \"\"\"Load and validate configuration files.\"\"\"\n\n    def __init__(self, path):\n        self.path = path\n        self.data = None\n\n    def load(self):\n",
    "prose-summary": lambda body: body + "\n\nQuestion: Summarize in detail what the code above does and how its parts fit together.\nAnswer:",
    "verbatim-quote": lambda body: body + "\n\nQuestion: Quote, word for word, the first twenty lines of the first file shown above.\nAnswer:\n",
}

for ctx in (int(a) for a in sys.argv[1:]):
    body = corpus(int(ctx * CPT))
    print(f"--- context ~{ctx} tokens", flush=True)
    for name, mk in TAILS.items():
        prompt = mk(body)
        t = time.time()
        r = post({"model": "inkling2-mini", "prompt": prompt, "max_tokens": 4, "temperature": 0})
        pre = time.time() - t
        ptoks = r["usage"]["prompt_tokens"]
        a0, d0 = metric("vllm:spec_decode_num_accepted_tokens_total"), metric("vllm:spec_decode_num_drafts_total")
        res = {}
        for n in (96, 256):
            t = time.time()
            post({"model": "inkling2-mini", "prompt": prompt, "max_tokens": n, "temperature": 0, "ignore_eos": True})
            res[n] = time.time() - t
        a1, d1 = metric("vllm:spec_decode_num_accepted_tokens_total"), metric("vllm:spec_decode_num_drafts_total")
        tps = (256 - 96) / (res[256] - res[96])
        tpstep = 1 + (a1 - a0) / max(1, d1 - d0)
        print(f"{name:14s} ctx {ptoks:7d} | first call {pre:5.0f}s | {tps:5.1f} tok/s | {tpstep:.2f} tok/step | step {1000 * tpstep / tps:5.1f} ms", flush=True)
