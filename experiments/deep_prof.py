"""One diverse long prompt (cold prefill) then a 200-token decode, to read per-phase PROF lines at depth. Usage: deep_prof.py CONTEXT_TOKENS"""
import glob, json, os, random, sys, time, urllib.request

URL = f"http://localhost:{os.environ.get('PORT', '8000')}/v1/completions"
files = sorted(glob.glob("/work/vllm/vllm/**/*.py", recursive=True)) + sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True))
random.Random(1234).shuffle(files)
n_chars = int(int(sys.argv[1]) * 3.0)
parts, n = [f"# run {random.random()}\n"], 0
for f in files:
    try:
        s = open(f, errors="ignore").read()
    except OSError:
        continue
    parts.append(f"\n# ==== {f}\n" + s)
    n += len(s)
    if n >= n_chars:
        break
body = "".join(parts)[:n_chars]
prompt = body + "\n# ==== /work/new_module.py\nimport os\nimport sys\nimport json\n\n\nclass ConfigLoader:\n    \"\"\"Load and validate configuration files.\"\"\"\n\n    def __init__(self, path):\n        self.path = path\n        self.data = None\n\n    def load(self):\n"


def post(mt):
    r = urllib.request.Request(URL, json.dumps({"model": "inkling2-mini", "prompt": prompt, "max_tokens": mt, "temperature": 0, "ignore_eos": True}).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=7200))["usage"]


t = time.time()
print("prefill:", post(4), f"{time.time() - t:.0f}s", flush=True)
t = time.time()
print("decode 200:", post(200), f"{time.time() - t:.0f}s", flush=True)
