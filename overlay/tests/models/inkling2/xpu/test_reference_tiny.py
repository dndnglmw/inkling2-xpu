# SPDX-License-Identifier: Apache-2.0
"""Synthetic tiny-config tests for the Inkling2 XPU reference implementation."""

import os
import sys

import torch

sys.path.insert(0, os.path.expanduser("~/work/vllm"))
from vllm.models.inkling2.xpu.reference import RefConfig, Inkling2Reference  # noqa: E402


def tiny_config() -> RefConfig:
    return RefConfig(
        vocab_size=512,
        unpadded_vocab_size=500,
        hidden_size=64,
        num_layers=12,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        swa_heads=4,
        swa_kv_heads=2,
        swa_head_dim=16,
        d_rel=4,
        rel_extent=32,
        sliding_window=32,
        sconv_kernel=4,
        rms_eps=1e-6,
        local_layer_ids={0, 1, 2, 4, 5, 6, 7, 8, 10, 11} - {5},
        log_scaling_n_floor=128,
        log_scaling_alpha=0.1,
        qk_clip=22.627416997969522,
        rel_clip=8.0,
        block_size=3,
        gate_softcap=4.0,
        up_softcap=25.0,
        final_softcap=30.0,
        n_experts=6,
        top_k=2,
        inter_size=24,
        route_scale=8.0,
        use_gate_bias=True,
        core_start=3,
        core_end=9,
        num_passes=3,
        gate_rank=8,
    )


def tiny_weights(cfg: RefConfig, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def randn(*shape, scale=0.02):
        return torch.randn(*shape, generator=g) * scale

    w: dict[str, torch.Tensor] = {}
    D, V = cfg.hidden_size, cfg.vocab_size
    E, I = cfg.n_experts, cfg.inter_size
    H, KV, HD = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    DR = cfg.d_rel
    K = cfg.sconv_kernel
    R = cfg.gate_rank

    w["model.llm.embed.weight"] = randn(V, D)
    w["model.llm.embed_norm.weight"] = 1.0 + randn(D)
    w["model.llm.norm.weight"] = 1.0 + randn(D)
    w["model.llm.unembed.weight"] = randn(V, D)
    w["model.llm.attn_res.output_query"] = randn(D)
    for i in range(cfg.num_layers - 1):
        n_q = 1 if i + 1 in cfg.local_layer_ids else 3
        w[f"model.llm.attn_res.attn_queries.{i}"] = randn(n_q * D)
    for i in range(cfg.num_layers):
        w[f"model.llm.attn_res.mlp_queries.{i}"] = randn(D)
    n_core = cfg.core_end - cfg.core_start
    w["model.llm.middle_cycle_cross_pass_queries"] = randn(2 * n_core, D)
    w["model.llm.middle_cycle_cross_pass_kv_queries"] = randn(2 * n_core, D)

    for i in range(cfg.num_layers):
        local = i in cfg.local_layer_ids
        pre = f"model.llm.layers.{i}."
        kv_ch = KV * HD
        w[pre + "attn.wq_du.weight"] = randn(H * HD, D)
        w[pre + "attn.wk_dv.weight"] = randn(kv_ch, D)
        w[pre + "attn.wv_dv.weight"] = randn(kv_ch, D)
        w[pre + "attn.wr_du.weight"] = randn(H * DR, D)
        w[pre + "attn.wo_ud.weight"] = randn(D, H * HD)
        w[pre + "attn.gate_attn.weight"] = randn(H * HD, D)
        w[pre + "attn.attn_sink_proj.weight"] = randn(H, D)
        w[pre + "attn.q_l2_clip.scale"] = randn(H, HD).abs() + 0.5
        w[pre + "attn.rel_l2_clip.scale"] = randn(H, DR).abs() + 0.5
        extent = cfg.sliding_window if local else cfg.rel_extent
        w[pre + "attn.rel_logits_proj.proj"] = randn(DR, extent)
        if local:
            w[pre + "attn.k_sconv.weight"] = randn(kv_ch, 1, K)
            w[pre + "attn.v_sconv.weight"] = randn(kv_ch, 1, K)
            w[pre + "attn_norm.weight"] = 1.0 + randn(D)
        else:
            w[pre + "attn.k_sconv.static_w"] = randn(K, kv_ch)
            w[pre + "attn.k_sconv.weight_proj"] = randn(KV * K, D)
            w[pre + "attn.v_sconv.static_w"] = randn(K, kv_ch)
            w[pre + "attn.v_sconv.weight_proj"] = randn(KV * K, D)
            for s in ("q", "k", "v"):
                w[pre + f"{s}_input_norm.weight"] = 1.0 + randn(D)
        w[pre + "attn_sconv.weight"] = randn(D, 1, K)
        w[pre + "mlp_sconv.weight"] = randn(D, 1, K)
        w[pre + "mlp_norm.weight"] = 1.0 + randn(D)
        w[pre + "mlp.gate.weight"] = randn(E, D)
        w[pre + "mlp.gate.bias"] = randn(E)
        w[pre + "mlp.experts.w13_weight"] = randn(E, 2 * I, D)
        w[pre + "mlp.experts.w2_weight"] = randn(E, D, I)
        if i >= cfg.core_start and i < cfg.core_end:
            for ctrl in ("attn", "k", "v", "mlp"):
                w[pre + f"controllers.{ctrl}.down_weight"] = randn(R, 2 * D)
                w[pre + f"controllers.{ctrl}.gain_weight"] = randn(D, R)
                w[pre + f"controllers.{ctrl}.forget_weight"] = randn(D, R)
    return w


def main() -> None:
    torch.manual_seed(0)
    cfg = tiny_config()
    weights = tiny_weights(cfg)
    model = Inkling2Reference(cfg, weights)

    ids = torch.randint(0, cfg.unpadded_vocab_size, (7,))
    logits_prefill = model.prefill(ids)
    assert logits_prefill.shape == (7, cfg.unpadded_vocab_size)
    assert torch.isfinite(logits_prefill).all(), "non-finite logits in prefill"

    # decode must reproduce prefill's last-token logits exactly
    model.reset()
    for t in ids[:-1]:
        model.decode(t.item())
    logits_decode = model.decode(ids[-1].item())
    diff = (logits_decode[0] - logits_prefill[-1]).abs().max().item()
    scale = logits_prefill[-1].abs().max().item()
    print(f"decode-vs-prefill max |diff| = {diff:.3e} (logit scale {scale:.1f})")
    assert diff < 5e-2 * max(scale, 1.0), "decode/prefill mismatch"

    # generation smoke test
    model.reset()
    out = model.generate(ids[:3], max_new=6, eos=cfg.unpadded_vocab_size - 1)
    print("generated:", out.tolist())

    # pass-count sanity: messages must change the output vs a 1-pass run
    cfg1 = tiny_config()
    cfg1.num_passes = 2
    m1 = Inkling2Reference(cfg1, tiny_weights(cfg1))
    l1 = m1.prefill(ids)
    d = (l1[-1] - logits_prefill[-1]).abs().max().item()
    print(f"2-pass vs 3-pass logit diff = {d:.3f} (should be nonzero)")
    assert d > 1e-3

    print("OK")


if __name__ == "__main__":
    main()