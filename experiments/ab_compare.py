"""Compare two ab_quality captures (fp8 vs bf16 KV): top-1 agreement, KL over shared
top-20 mass, greedy text match, needle recall. Usage: ab_compare.py [fp8] [bf16]"""
import json, math, sys

A, B = sys.argv[1] if len(sys.argv) > 1 else "fp8", sys.argv[2] if len(sys.argv) > 2 else "bf16"
a = json.load(open(f"/work/ab-quality-{A}.json"))
b = json.load(open(f"/work/ab-quality-{B}.json"))


def kl(d1, d2):
    sh = [t for t in d1 if t in d2]
    if not sh:
        return None
    pa = [math.exp(d1[t]) for t in sh]
    pb = [math.exp(d2[t]) for t in sh]
    sa, sb = sum(pa), sum(pb)
    return sum(v / sa * math.log((v / sa) / (w / sb)) for v, w in zip(pa, pb))


kls, agree, tot = [], 0, 0
for x, y in zip(a["short"], b["short"]):
    for d1, d2 in zip(x["top"], y["top"]):
        if not d1 or not d2:
            continue
        agree += max(d1, key=d1.get) == max(d2, key=d2.get)
        tot += 1
        k = kl(d1, d2)
        if k is not None:
            kls.append(k)
gmatch = sum(x["text"] == y["text"] for x, y in zip(a["short"], b["short"]))

print(f"== {A} vs {B} (reference) ==")
print(f"top-1 agreement: {agree}/{tot} positions")
if kls:
    print(f"KL({A}||{B}) over shared top-20: mean {sum(kls) / len(kls):.5f}  max {max(kls):.5f}  ({len(kls)} positions)")
print(f"greedy 48-token text exact match: {gmatch}/{len(a['short'])}")
for x, y in zip(a["needle"], b["needle"]):
    fa = x.get("found"); fb = y.get("found")
    print(f"needle {x['ctx']:>6}: {A}={'FOUND' if fa else 'MISSED' if fa is not None else 'ERR'}  "
          f"{B}={'FOUND' if fb else 'MISSED' if fb is not None else 'ERR'}  | {A} {x.get('ans')!r} | {B} {y.get('ans')!r}")