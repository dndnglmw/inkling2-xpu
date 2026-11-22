import json, math, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
kls, agree = [], 0
for x, y in zip(a["dist"], b["dist"]):
    sh = [t for t in x if t in y]
    px = [math.exp(x[t]) for t in sh]; py = [math.exp(y[t]) for t in sh]
    zx, zy = sum(px), sum(py)
    kls.append(sum((p / zx) * math.log((p / zx) / (q / zy)) for p, q in zip(px, py)))
    agree += max(x, key=x.get) == max(y, key=y.get)
print(f"prompts {len(kls)} | top-1 agree {agree}/{len(kls)} | mean KL(old||new) {sum(kls) / len(kls):.6f} | max KL {max(kls):.6f}")
for i, (g1, g2) in enumerate(zip(a["greedy"], b["greedy"])):
    n = next((k for k, (u, v) in enumerate(zip(g1, g2)) if u != v), len(g1))
    print(f"  greedy prompt {i}: identical for the first {n}/{len(g1)} tokens")
