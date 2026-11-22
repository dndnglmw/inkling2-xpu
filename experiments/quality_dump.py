"""Dump next-token top-40 logprobs for the 12 quality prompts (+ 24-token greedy for 3) to a json; compare two runs with compare_dumps.py."""
import json, os, re, sys
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
sys.path.insert(0, "/work/vllm")
from vllm import LLM, SamplingParams

out_path = sys.argv[1]
kv_dtype = os.environ.get("KVD", "auto")  # auto=bf16 (baseline) | fp8 (real e4m3 KV)
src = open("/work/vllm/tests/models/inkling2/xpu/quality_vs_llamacpp.py").read()
PROMPTS = eval(re.search(r"PROMPTS = (\[.*?\n\])", src, re.S).group(1))
llm = LLM(model="/work/models/inkling-xpu4", dtype="bfloat16", enforce_eager=True, tensor_parallel_size=1, max_model_len=4096,
          max_num_batched_tokens=2048, max_num_seqs=2, gpu_memory_utilization=0.9, enable_prefix_caching=False,
          limit_mm_per_prompt={"image": 0}, max_logprobs=40,
          kv_cache_dtype=None if kv_dtype == "auto" else kv_dtype)
res = {"dist": [], "greedy": []}
for p in PROMPTS:
    r = llm.generate([p], SamplingParams(max_tokens=1, temperature=0, logprobs=40), use_tqdm=False)[0].outputs[0].logprobs[0]
    res["dist"].append({str(t): l.logprob for t, l in r.items()})
for p in PROMPTS[:3]:
    res["greedy"].append(list(llm.generate([p], SamplingParams(max_tokens=24, temperature=0, ignore_eos=True), use_tqdm=False)[0].outputs[0].token_ids))
json.dump(res, open(out_path, "w"))
print("saved", out_path)
