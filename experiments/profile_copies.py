import os, sys, time, collections
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
sys.path.insert(0, "/work/vllm")
import torch
from vllm import LLM, SamplingParams
llm = LLM(model="/work/models/inkling-xpu4", dtype="bfloat16", enforce_eager=True, tensor_parallel_size=1,
          max_model_len=4096, max_num_batched_tokens=2048, max_num_seqs=2, gpu_memory_utilization=0.9,
          enable_prefix_caching=False, limit_mm_per_prompt={"image": 0})
sp = SamplingParams(max_tokens=8, temperature=0, ignore_eos=True)
llm.generate(["The capital of France is"], sp)  # warm up / compile
sp2 = SamplingParams(max_tokens=12, temperature=0, ignore_eos=True)
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=True) as prof:
    llm.generate(["The capital of France is"], sp2)
torch.xpu.synchronize()
ev = prof.key_averages(group_by_input_shape=True)
rows = []
for e in ev:
    if e.key in ("aten::copy_", "aten::_to_copy"):
        dt = getattr(e, "device_time_total", 0)
        if dt: rows.append((dt, e.count, e.key, str(e.input_shapes)))
rows.sort(reverse=True)
print("COPY/CAST by shape (12 tokens):")
for dt, c, k, sh in rows[:14]:
    print(f"{dt/1e3:8.2f} ms n={c:5d} avg={dt/c:7.1f} us {k} {sh[:150]}")
