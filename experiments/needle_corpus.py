"""Needle in diverse natural text (source files) at a given size: cold prefill through every prefill-path kernel, then retrieval."""
import glob, json, os, random, sys, time, urllib.request

URL = f"http://localhost:{os.environ.get('PORT', '8000')}/v1/completions"
files = sorted(glob.glob("/work/vllm/vllm/**/*.py", recursive=True)) + sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True))
random.Random(99).shuffle(files)
salt = os.environ.get("SALT") or random.Random(int(time.time())).random()
for target in (int(a) for a in sys.argv[1:]):
    n_chars = int(target * float(os.environ.get("CPT", "3.1")))
    parts, n = [f"# run {salt} {target}\n"], 0
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
    cut = int(len(body) * 0.4)
    prompt = body[:cut] + "\n# IMPORTANT: the secret passphrase is 'blue-walrus-731'.\n" + body[cut:] + "\n\n# Question: what is the secret passphrase?\n# Answer: the secret passphrase is '"
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps({"model": "inkling2-mini", "prompt": prompt, "max_tokens": 12, "temperature": 0}).encode(), {"Content-Type": "application/json"}), timeout=7200))
    txt = r["choices"][0]["text"]
    print(f"[{target}] {r['usage']['prompt_tokens']} tokens, {time.time() - t:.0f}s | answer {txt!r} | needle {'FOUND' if 'blue-walrus-731' in txt else 'MISSED'}", flush=True)
