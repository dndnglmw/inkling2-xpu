# SPDX-License-Identifier: Apache-2.0
"""Oracle trace: run the real checkpoint through transformers Inkling2 (non-loop).

The loop weights are ignored; the model becomes pass-0 of the looped model.
Saves per-token-norm checksums of embedding and each layer output plus final
logits, to bisect the pure-torch reference op by op.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.expanduser("~/work/vllm"))

from transformers import Inkling2TextConfig  # noqa: E402
from transformers.models.inkling2.modeling_inkling2 import Inkling2ForCausalLM  # noqa: E402
from safetensors import safe_open  # noqa: E402

MODEL_DIR = os.path.expanduser("~/work/models/inkling-bf16")
IDS = [371, 151451, 149879, 151194, 149884]

RENAMES = {
    "model.llm.embed.weight": "model.embed_tokens.weight",
    "model.llm.embed_norm.weight": "model.embed_tokens.embed_norm.weight",
    "model.llm.norm.weight": "model.norm.weight",
    "model.llm.unembed.weight": "lm_head.weight",
    "model.llm.attn_res.output_query": "model.final_input.query",
}


def rename(name: str) -> str | None:
    if name in RENAMES:
        return RENAMES[name]
    if ".attn_res.attn_queries." in name:
        i = int(name.rsplit(".", 1)[1])
        return f"model.layers.{i + 1}.attention_input.query"
    if ".attn_res.mlp_queries." in name:
        i = int(name.rsplit(".", 1)[1])
        return f"model.layers.{i}.mlp_input.query"
    if not name.startswith("model.llm.layers."):
        return None
    rest = name[len("model.llm.layers."):]
    li, tail = rest.split(".", 1)
    p = f"model.layers.{li}.self_attn."
    m = {
        "attn.wq_du.weight": p + "q_proj.weight",
        "attn.wk_dv.weight": p + "k_proj.weight",
        "attn.wv_dv.weight": p + "v_proj.weight",
        "attn.wr_du.weight": p + "r_proj.weight",
        "attn.wo_ud.weight": p + "o_proj.weight",
        "attn.gate_attn.weight": p + "gate_proj.weight",
        "attn.attn_sink_proj.weight": p + "sink_proj.weight",
        "attn.q_l2_clip.scale": p + "q_l2_clip.scale",
        "attn.rel_l2_clip.scale": p + "rel_l2_clip.scale",
        "attn.rel_logits_proj.proj": p + "rel_logits_proj.proj",
        "attn.k_sconv.weight": p + "k_sconv.conv1d.weight",
        "attn.v_sconv.weight": p + "v_sconv.conv1d.weight",
        "attn.k_sconv.static_w": p + "k_sconv.static_weight",
        "attn.v_sconv.static_w": p + "v_sconv.static_weight",
        "attn.k_sconv.weight_proj": p + "k_sconv.weight_proj.weight",
        "attn.v_sconv.weight_proj": p + "v_sconv.weight_proj.weight",
        "attn_norm.weight": f"model.layers.{li}.input_layernorm.weight",
        "q_input_norm.weight": f"model.layers.{li}.q_input_layernorm.weight",
        "k_input_norm.weight": f"model.layers.{li}.k_input_layernorm.weight",
        "v_input_norm.weight": f"model.layers.{li}.v_input_layernorm.weight",
        "attn_sconv.weight": f"model.layers.{li}.post_attention_convolution.conv1d.weight",
        "mlp_sconv.weight": f"model.layers.{li}.post_mlp_convolution.conv1d.weight",
        "mlp_norm.weight": f"model.layers.{li}.post_attention_layernorm.weight",
        "mlp.gate.weight": f"model.layers.{li}.mlp.gate.weight",
        "mlp.gate.bias": f"model.layers.{li}.mlp.gate.e_score_correction_bias",
        "mlp.experts.w13_weight": f"model.layers.{li}.mlp.experts.gate_up_proj",
        "mlp.experts.w2_weight": f"model.layers.{li}.mlp.experts.down_proj",
    }
    return m.get(tail)


def main() -> None:
    cfg_dict = json.loads(open(f"{MODEL_DIR}/config.json").read())["text_config"]
    cfg_dict.pop("loop_config", None)
    cfg = Inkling2TextConfig(**cfg_dict)
    cfg.model_type = "inkling2_text"
    cfg._attn_implementation = "eager"

    torch.set_default_dtype(torch.bfloat16)
    model = Inkling2ForCausalLM(cfg).to(torch.bfloat16).eval()
    # fp32 parts: router and AttnRes queries (HF math casts activations to float)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("mlp.gate.weight") or "attn_res" in name or name.endswith(("attention_input.query", "mlp_input.query", "final_input.query")):
                p.data = p.data.float()
        for name, b in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                b.data = b.data.float()
    index = json.loads(open(f"{MODEL_DIR}/model.safetensors.index.json").read())["weight_map"]
    loaded: dict[str, str] = {}
    sd: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for shard, names in _group(index):
            with safe_open(f"{MODEL_DIR}/{shard}", framework="pt") as f:
                for n in names:
                    dst = rename(n)
                    if dst is None:
                        continue
                    sd[dst] = f.get_tensor(n).to(torch.bfloat16)
                    loaded[n] = dst
    res = model.load_state_dict(sd, strict=False)
    missing = [k for k in res.missing_keys if "rotary" not in k]
    if missing:
        print("MISSING:", missing[:10], f"(total {len(missing)})")
        sys.exit(1)

    ids = torch.tensor([IDS])
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    trace = {
        "ids": IDS,
        "layers": [[float(t.abs().sum()), float(t.square().sum())] for t in hs],
        "logits_top": torch.topk(out.logits[0, -1].float(), 16).values.tolist(),
        "logits_top_idx": torch.topk(out.logits[0, -1].float(), 16).indices.tolist(),
    }
    json.dump(trace, open("/tmp/hf_trace.json", "w"))
    print("hf trace saved; embed norm:", trace["layers"][0], "layer23:", trace["layers"][24])
    print("top logits:", list(zip(trace["logits_top_idx"], [round(v, 2) for v in trace["logits_top"]])))


def _group(weight_map):
    shards: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        shards.setdefault(shard, []).append(name)
    for shard in sorted(shards):
        yield shard, shards[shard]


if __name__ == "__main__":
    main()
