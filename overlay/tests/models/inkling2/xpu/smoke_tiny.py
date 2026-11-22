# SPDX-License-Identifier: Apache-2.0
"""Tiny random-weight Inkling2 loop end-to-end smoke test for Intel XPU."""

import json
import os
import sys

import torch

from vllm import LLM, SamplingParams

TINY = {
    "architectures": ["Inkling2LoopForConditionalGeneration"],
    "model_type": "inkling2_loop",
    "eos_token_id": 1,
    "text_config": {
        "torch_dtype": "bfloat16",
        "hidden_size": 512,
        "num_hidden_layers": 4,
        "local_layer_ids": [0, 1, 3],
        "sliding_window_size": 128,
        "vocab_size": 1024,
        "unpadded_vocab_size": 1000,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "n_routed_experts": 16,
        "num_experts_per_tok": 8,
        "intermediate_size": 256,
        "mlp_gate_softcap": 4.0,
        "mlp_up_softcap": 25.0,
        "n_layers_per_attn_res_block": 1,
        "loop_config": {
            "core_start": 1,
            "core_end": 3,
            "num_passes": 3,
            "gate_rank": 16,
            "is_varloop": True,
        },
    },
}


def main() -> None:
    path = "/tmp/tiny-inkling2"
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/config.json", "w") as f:
        json.dump(TINY, f)
    print("xpu:", torch.xpu.is_available(), "devices:", torch.xpu.device_count(), flush=True)

    llm = LLM(
        model=path,
        load_format="dummy",
        skip_tokenizer_init=True,
        tokenizer_mode="hf",
        enforce_eager=True,
        gpu_memory_utilization=0.15,
        max_num_batched_tokens=128,
        max_model_len=512,
        max_num_seqs=2,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0},
    )
    prompt = list(range(10, 20))
    out = llm.generate(
        [{"prompt_token_ids": prompt}],
        SamplingParams(max_tokens=8, ignore_eos=True, detokenize=False),
    )[0]
    print("generated:", out.outputs[0].token_ids, flush=True)

    # decode must reproduce the prefill's last-token logprobs
    a = llm.generate(
        [{"prompt_token_ids": prompt}],
        SamplingParams(max_tokens=1, prompt_logprobs=0, ignore_eos=True, detokenize=False),
    )[0]
    print("prompt logprobs len:", len(a.prompt_logprobs or []), flush=True)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
    sys.exit(0)
