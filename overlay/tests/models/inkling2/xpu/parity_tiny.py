# SPDX-License-Identifier: Apache-2.0
"""Parity: tiny random checkpoint, pure-torch reference vs vLLM XPU path."""
import json, os, sys
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/work/vllm")
from test_reference_tiny import tiny_config, tiny_weights
from vllm.models.inkling2.xpu.reference import Inkling2Reference

SCALE = float(os.environ.get("SCALE", "4"))
PATH = "/tmp/tiny-parity"
cfg = tiny_config()
cfg.num_passes = int(os.environ.get("PASSES", cfg.num_passes))
base = json.load(open("/work/models/inkling-bf16/config.json"))
base.pop("quantization_config", None)
base.pop("mtp_config", None)
t = base["text_config"]
t.update(
    hidden_size=cfg.hidden_size, num_hidden_layers=cfg.num_layers, vocab_size=cfg.vocab_size,
    unpadded_vocab_size=cfg.unpadded_vocab_size, num_attention_heads=cfg.num_heads,
    num_key_value_heads=cfg.num_kv_heads, head_dim=cfg.head_dim, d_rel=cfg.d_rel,
    rel_extent=cfg.rel_extent, local_layer_ids=sorted(cfg.local_layer_ids),
    log_scaling_n_floor=cfg.log_scaling_n_floor, swa_head_dim=cfg.swa_head_dim,
    swa_num_attention_heads=cfg.swa_heads, swa_num_key_value_heads=cfg.swa_kv_heads,
    sliding_window_size=cfg.sliding_window, n_layers_per_attn_res_block=cfg.block_size,
    n_routed_experts=cfg.n_experts, num_experts_per_tok=cfg.top_k,
    intermediate_size=cfg.inter_size, model_max_length=2048,
)
t["loop_config"].update(core_start=cfg.core_start, core_end=cfg.core_end,
                        num_passes=cfg.num_passes, gate_rank=cfg.gate_rank)
os.makedirs(PATH, exist_ok=True)
json.dump(base, open(f"{PATH}/config.json", "w"))
import shutil
for f in ("tokenizer.json", "chat_template.jinja"):
    shutil.copy(f"/work/models/inkling-bf16/{f}", PATH)
w = tiny_weights(cfg, seed=1)
# bigger matmul weights so logits carry signal
for k, v in w.items():
    if v.ndim >= 2 and "sconv" not in k and "queries" not in k:
        w[k] = v * SCALE
F32 = ("attn_queries", "mlp_queries", "output_query", "mlp.gate.bias", "cross_pass_kv_queries", "cross_pass_queries")
w = {k: (v.float() if any(x in k for x in F32) else v.to(torch.bfloat16)).contiguous() for k, v in w.items()}
save_file(w, f"{PATH}/model.safetensors")
json.dump({"weight_map": {k: "model.safetensors" for k in w}, "metadata": {}},
          open(f"{PATH}/model.safetensors.index.json", "w"))

ids = torch.arange(10, 30)
ref = Inkling2Reference(cfg, {k: v.float() for k, v in w.items()})
REF = []
_orig = ref._layer
def _wrap(slot, stream, blocks, positions, messages):
    new_stream, nb = _orig(slot, stream, blocks, positions, messages)
    REF.append(new_stream.float().cpu())
    return new_stream, nb
ref._layer = _wrap
RA, RM, RO = [], [], []
_oa = ref._attention
def _wa(slot, inputs, positions):
    o = _oa(slot, inputs, positions); RA.append(o.float().cpu()); return o
ref._attention = _wa
_om = ref._moe
def _wm(slot, x):
    RM.append(x.float().cpu()); o = _om(slot, x); RO.append(o.float().cpu()); return o
ref._moe = _wm
ref_logits = ref.prefill(ids)[-1].float()
ref_lp = torch.log_softmax(ref_logits, -1)

import vllm.models.inkling2.xpu.model as xm
from vllm import LLM, SamplingParams
llm = LLM(model=PATH, enforce_eager=True, gpu_memory_utilization=0.15,
          max_num_batched_tokens=128, max_model_len=512, max_num_seqs=2, dtype="bfloat16",
          enable_prefix_caching=False, limit_mm_per_prompt={"image": 0},
          max_logprobs=cfg.unpadded_vocab_size)
xm.DEBUG_DUMP = []
out = llm.generate([{"prompt_token_ids": ids.tolist()}],
                   SamplingParams(max_tokens=1, logprobs=cfg.unpadded_vocab_size, detokenize=False))[0]
lp = out.outputs[0].logprobs[0]
x = torch.full((cfg.unpadded_vocab_size,), float("nan"))
for tid, l in lp.items():
    if tid < cfg.unpadded_vocab_size:
        x[tid] = l.logprob
m = ~torch.isnan(x)
corr = torch.corrcoef(torch.stack([x[m], ref_lp[: cfg.unpadded_vocab_size][m]]))[0, 1]
print("n compared", int(m.sum()), "corr", float(corr))
print("ref top5", ref_lp.topk(5).indices.tolist(), "vllm top5", x.nan_to_num(-1e9).topk(5).indices.tolist())
print("max abs diff", float((x[m] - ref_lp[: cfg.unpadded_vocab_size][m]).abs().max()))

n = len(ids)
dumps = [d for d in xm.DEBUG_DUMP if d["stream"].shape[0] == n]
print("visits ref", len(REF), "vllm", len(dumps))
for r, d in zip(REF, dumps):
    a, b = r, d["stream"]
    c = torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1]
    print(f"layer {d['i']:2d} pass {d['pass']} stream corr {float(c):.5f} rel_err {float((a-b).norm()/a.norm()):.4f}")

def cc(a, b):
    return float(torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1])
for k, d in enumerate(dumps):
    print(f"visit {k} layer {d['i']}: attn_partial {cc(RA[k], d['attn_partial']):.5f} mlp_in {cc(RM[k], d['mlp_in']):.5f} moe {cc(RO[k], d['moe']):.5f}")
k = 5
e = ((RO[k] - dumps[k]["moe"]).norm(dim=-1) / RO[k].norm(dim=-1))
print("moe visit5 per-token rel err", [round(float(x), 3) for x in e])

# ---- decode parity: per-layer streams for the first decode token ----
xm.DEBUG_DUMP = []
out = llm.generate([{"prompt_token_ids": ids.tolist()}],
                   SamplingParams(max_tokens=2, ignore_eos=True, detokenize=False))[0]
t0 = out.outputs[0].token_ids[0]
dd = [d for d in xm.DEBUG_DUMP if d["stream"].shape[0] == 1]
print("decode visits vllm", len(dd))
dd = dd[-len(REF):]
ref.reset(); ref.prefill(ids)
REF.clear(); RA.clear(); RM.clear(); RO.clear()
ref.decode(int(t0))
for k, d in enumerate(dd):
    print(f"dec visit {k:2d} layer {d['i']:2d} pass {d['pass']}: attn_partial {cc(RA[k], d['attn_partial']):.5f} mlp_in {cc(RM[k], d['mlp_in']):.5f} moe {cc(RO[k], d['moe']):.5f} stream {cc(REF[k], d['stream']):.5f}")

# ---- self-consistency: decode of t0 vs last row of prefill(ids+[t0]) ----
xm.DEBUG_DUMP = []
llm.generate([{"prompt_token_ids": ids.tolist() + [int(t0)]}],
             SamplingParams(max_tokens=1, ignore_eos=True, detokenize=False))
pf = [d for d in xm.DEBUG_DUMP if d["stream"].shape[0] == len(ids) + 1][-24:]
for k, (a, b) in enumerate(zip(pf, dd)):
    print(f"self visit {k:2d} layer {a['i']:2d} pass {a['pass']}: attn_partial {cc(a['attn_partial'][-1], b['attn_partial'][0]):.5f} "
          f"mlp_in {cc(a['mlp_in'][-1], b['mlp_in'][0]):.5f} moe {cc(a['moe'][-1], b['moe'][0]):.5f} stream {cc(a['stream'][-1], b['stream'][0]):.5f}")
