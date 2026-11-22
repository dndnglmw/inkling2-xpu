# SPDX-License-Identifier: Apache-2.0
"""Cross-check the BF16 dequantized experts against the llama.cpp Q8_0 GGUF."""

import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.expanduser("~/work/llama.cpp/gguf-py"))
import gguf  # noqa: E402

GGUF = os.path.expanduser("~/work/models/inkling-gguf/Inkling-2-Mini-Q8_0.gguf")
BF16 = os.path.expanduser("~/work/models/inkling-bf16")


def gguf_tensor(reader, name: str) -> tuple[np.ndarray, int]:
    t = next(t for t in reader.tensors if t.name == name)
    ne0, ne1, ne2 = t.shape
    if t.tensor_type != gguf.GGMLQuantizationType.Q8_0:
        raise SystemExit(f"{name}: unhandled qtype {t.tensor_type}")
    bs, ts = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q8_0]
    nbytes = (ne0 // bs) * ts * ne1
    out = np.empty((ne2, ne1, ne0), dtype=np.float32)
    arr = np.frombuffer(t.data.tobytes(), dtype=np.uint8)
    for e in range(ne2):
        chunk = arr[e * nbytes:(e + 1) * nbytes]
        out[e] = gguf.quants.dequantize(chunk, gguf.GGMLQuantizationType.Q8_0).reshape(ne1, ne0)
    return out, ne2


def main() -> None:
    reader = gguf.GGUFReader(GGUF)
    index = json.load(open(f"{BF16}/model.safetensors.index.json"))["weight_map"]

    def load_bf16(name: str) -> torch.Tensor:
        with safe_open(f"{BF16}/{index[name]}", framework="pt") as f:
            return f.get_tensor(name).float()

    for layer, kind in ((5, "mxfp8"), (0, "nvfp4")):
        print(f"=== layer {layer} ({kind}) ===")
        gu, _ = gguf_tensor(reader, f"blk.{layer}.ffn_gate_up_exps.weight")
        down, _ = gguf_tensor(reader, f"blk.{layer}.ffn_down_exps.weight")
        w13 = load_bf16(f"model.llm.layers.{layer}.mlp.experts.w13_weight")  # [E, 2I, D]
        w2 = load_bf16(f"model.llm.layers.{layer}.mlp.experts.w2_weight")  # [E, D, I]
        e = 0
        a = torch.from_numpy(gu[e])  # [2I, D]
        b = w13[e]  # [2I, D]
        I = b.shape[0] // 2
        # try both fused layouts: chunked [gate; up] and interleaved (gate,up) pairs
        chunked = (a - b).norm() / b.norm()
        # interleaved interpretation of the gguf rows -> deinterleave to compare
        inter = a.view(I, 2, -1)
        deint = torch.cat([inter[:, 0], inter[:, 1]], 0)
        interleaved = (deint - b).norm() / b.norm()
        print(f"  gate_up: rel(chunked)={chunked:.5f} rel(interleaved)={interleaved:.5f}")
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
        print(f"  gate_up cos(flat)={cos:.6f} |gguf|={a.norm():.2f} |bf16|={b.norm():.2f}")
        d = torch.from_numpy(down[e])  # [D, I]
        rel = (d - w2[e]).norm() / w2[e].norm()
        cos = torch.nn.functional.cosine_similarity(d.flatten(), w2[e].flatten(), dim=0)
        print(f"  down: rel={rel:.5f} cos={cos:.6f} |gguf|={d.norm():.2f} |bf16|={w2[e].norm():.2f}")
        print(f"  sample gate_up [0,:4] gguf={a[0, :4].tolist()} bf16={b[0, :4].tolist()}")
        print(f"  sample down    [0,:4] gguf={d[0, :4].tolist()} bf16={w2[e][0, :4].tolist()}")


if __name__ == "__main__":
    main()
