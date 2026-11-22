"""Cold prefill tok/s on the live server with diverse natural text (source files; unique per run so the prefix cache cannot hit).
Usage: prefill_bench.py TARGET_TOKENS [...]   (PORT env, default 8000). Chars/token calibrated at ~3.3 for this corpus, then the
server-reported prompt_tokens is what is divided by time."""
import glob, json, os, random, sys, time, urllib.request

URL = f"http://localhost:{os.environ.get('PORT', '8000')}/v1/completions"
CPT = float(os.environ.get("CPT", "3.3"))
files = sorted(glob.glob("/work/vllm/vllm/**/*.py", recursive=True)) + sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True))
random.Random(1234).shuffle(files)  # fixed corpus order: identical text across builds (only the salt line differs)
rng = random.Random(int(time.time()))  # salt only


def corpus(n_chars):
    out, n = [f"# run {rng.random()}\n"], 0
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


for target in (int(a) for a in sys.argv[1:]):
    prompt = corpus(int(target * CPT))
    body = json.dumps({"model": "inkling2-mini", "prompt": prompt, "max_tokens": 1, "temperature": 0}).encode()
    t = time.time()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=7200))
    except urllib.error.HTTPError as e:
        print(target, "HTTP", e.code, e.read()[:200]); continue
    dt = time.time() - t
    n = r["usage"]["prompt_tokens"]
    print(f"prefill {n:7d} tokens in {dt:6.1f}s = {n / dt:6.0f} tok/s   (target {target}, chars/token actual {len(prompt) / n:.2f})", flush=True)
