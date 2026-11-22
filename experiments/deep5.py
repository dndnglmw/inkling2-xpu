"""Real-text acceptance + step cost over many tails on a diverse long context (source-file corpus).
Usage: deep5.py SEED CONTEXT_TOKENS [...]    (PORT env; CPT env chars/token for sizing, default 3.0)
Per tail: tokens/step (spec metrics) and step ms (marginal timing 256 vs 96 tokens). Summary: mean/min/max/std over tails."""
import glob, json, os, random, statistics as st, sys, time, urllib.request

URL = f"http://localhost:{os.environ.get('PORT', '8000')}"
CPT = float(os.environ.get("CPT", "3.0"))
SEED = int(sys.argv[1])
files = sorted(glob.glob("/work/vllm/vllm/**/*.py", recursive=True)) + sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True))
random.Random(SEED).shuffle(files)
SALT = random.Random(int(time.time())).random()


def post(p):
    r = urllib.request.Request(URL + "/v1/completions", json.dumps(p).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=7200))


def metric(prefix):
    t = urllib.request.urlopen(URL + "/metrics").read().decode()
    return sum(float(l.split()[-1]) for l in t.splitlines() if l.startswith(prefix) and "created" not in l.split()[0])


def corpus(n_chars):
    out, n = [f"# run {SALT} seed {SEED} {n_chars}\n"], 0
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


Q = "\n\nQuestion: {}\nAnswer:"
TAILS = {
    "code-class": lambda b: b + "\n# ==== /work/new_module.py\nimport os\nimport sys\nimport json\n\n\nclass ConfigLoader:\n    \"\"\"Load and validate configuration files.\"\"\"\n\n    def __init__(self, path):\n        self.path = path\n        self.data = None\n\n    def load(self):\n",
    "code-func": lambda b: b + "\n# ==== /work/cli.py\nimport argparse\n\n\ndef parse_args(argv=None):\n    \"\"\"Parse command line arguments for the batch converter.\"\"\"\n    parser = argparse.ArgumentParser(description=\"batch converter\")\n",
    "summary": lambda b: b + Q.format("Summarize in detail what the code above does and how its parts fit together."),
    "explain": lambda b: b + Q.format("Explain step by step what the most complex function in the code above does."),
    "quote": lambda b: b + Q.format("Quote, word for word, the first twenty lines of the first file shown above.") + "\n",
    "list-imports": lambda b: b + Q.format("Which Python modules are imported most often in the code above? List ten of them with a short note on what each is used for."),
    "refactor": lambda b: b + Q.format("Rewrite the first function in the first file above with full type hints and a docstring."),
    "write-tests": lambda b: b + Q.format("Write pytest unit tests for the first class in the first file above."),
}


def measure(prompt):
    t = time.time()
    r = post({"model": "inkling2-mini", "prompt": prompt, "max_tokens": 4, "temperature": 0})
    pre = time.time() - t
    a0, d0 = metric("vllm:spec_decode_num_accepted_tokens_total"), metric("vllm:spec_decode_num_drafts_total")
    res = {}
    for n in (96, 256):
        t = time.time()
        post({"model": "inkling2-mini", "prompt": prompt, "max_tokens": n, "temperature": 0, "ignore_eos": True})
        res[n] = time.time() - t
    a1, d1 = metric("vllm:spec_decode_num_accepted_tokens_total"), metric("vllm:spec_decode_num_drafts_total")
    tps = 160 / (res[256] - res[96])
    tpstep = 1 + (a1 - a0) / max(1, d1 - d0)
    return r["usage"]["prompt_tokens"], pre, tps, tpstep


for ctx in (int(a) for a in sys.argv[2:]):
    body = corpus(int(ctx * CPT))
    print(f"--- seed {SEED} context ~{ctx} tokens", flush=True)
    rows = []
    for name, mk in TAILS.items():
        ptoks, pre, tps, tpstep = measure(mk(body))
        rows.append((tpstep, 1000 * tpstep / tps, tps))
        print(f"{name:13s} ctx {ptoks:7d} | first {pre:4.0f}s | {tps:5.1f} tok/s | {tpstep:.2f} tok/step | step {1000 * tpstep / tps:5.1f} ms", flush=True)
    tp = [r[0] for r in rows]
    print(f"SUMMARY ctx~{ctx}: tok/step mean {st.mean(tp):.2f} min {min(tp):.2f} max {max(tp):.2f} std {st.pstdev(tp):.2f} | step ms mean {st.mean(r[1] for r in rows):.1f} | tok/s mean {st.mean(r[2] for r in rows):.1f}", flush=True)
