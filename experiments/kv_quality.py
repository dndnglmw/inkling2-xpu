import os, sys, math, json
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
sys.path.insert(0, "/work/vllm")
import torch
from vllm import LLM, SamplingParams
from vllm.models.inkling2.xpu import kv_sim
llm = LLM(model="/work/models/inkling-xpu4", dtype="bfloat16", enforce_eager=True, tensor_parallel_size=1,
          max_model_len=20000, max_num_batched_tokens=2048, max_num_seqs=2, gpu_memory_utilization=0.9,
          enable_prefix_caching=False, limit_mm_per_prompt={"image": 0}, max_logprobs=40)
sys.path.insert(0, "/work/vllm/tests/models/inkling2/xpu")
import re
src = open("/work/vllm/tests/models/inkling2/xpu/quality_vs_llamacpp.py").read()
PROMPTS = eval(re.search(r"PROMPTS = (\[.*?\n\])", src, re.S).group(1))
filler = "The quick brown fox jumps over the lazy dog near the river bank while the sun sets slowly. "
long_prompts = []
for n, pos in ((4000, 0.5), (14000, 0.3)):
    reps = n // 20
    body = filler * int(reps * pos) + "IMPORTANT: the secret passphrase is 'purple-giraffe-42'. " + filler * (reps - int(reps * pos))
    long_prompts.append(body + "\n\nQuestion: What is the secret passphrase?\nAnswer: The secret passphrase is '")
sp = SamplingParams(max_tokens=1, temperature=0, logprobs=40)
def dist(prompts):
    out = []
    for p in prompts:
        r = llm.generate([p], sp, use_tqdm=False)[0].outputs[0].logprobs[0]
        out.append({t: l.logprob for t, l in r.items()})
    return out
def greedy(p, n=24):
    return llm.generate([p], SamplingParams(max_tokens=n, temperature=0, ignore_eos=True), use_tqdm=False)[0].outputs[0].token_ids
allp = PROMPTS + long_prompts
kv_sim.set_mode("off")
base = dist(allp); base_g = [greedy(p) for p in (PROMPTS[0], PROMPTS[1], long_prompts[0])]
def kl(a, b):
    sh = [t for t in a if t in b]
    pa = torch.tensor([math.exp(a[t]) for t in sh]); pb = torch.tensor([math.exp(b[t]) for t in sh])
    pa, pb = pa / pa.sum(), pb / pb.sum()
    return float((pa * (pa / pb).log()).sum())
print("mode          mean KL(base||mode)  max KL   top1 agree   greedy24 match   needle(4K/14K)")
for mode in ("fp8", "rtn8", "tq4", "k8v4", "k4v4", "rtn4", "tq3"):
    kv_sim.set_mode(mode)
    d = dist(allp)
    kls = [kl(a, b) for a, b in zip(base, d)]
    agree = sum(max(a, key=a.get) == max(b, key=b.get) for a, b in zip(base, d))
    gm = [greedy(p) for p in (PROMPTS[0], PROMPTS[1], long_prompts[0])]
    match = sum(sum(x == y for x, y in zip(a, b)) for a, b in zip(base_g, gm)) / sum(len(a) for a in base_g)
    needle = [llm.generate([p], SamplingParams(max_tokens=6, temperature=0), use_tqdm=False)[0].outputs[0].text for p in long_prompts]
    ok = sum("purple" in t for t in needle)
    print(f"{mode:12s}  {sum(kls)/len(kls):10.5f}  {max(kls):9.5f}  {agree:3d}/{len(allp)}      {match*100:5.1f}%          {ok}/2", flush=True)
kv_sim.set_mode("off")
