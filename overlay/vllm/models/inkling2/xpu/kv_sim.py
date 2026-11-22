# SPDX-License-Identifier: Apache-2.0
"""Simulated KV-cache quantisation (round-trip) for quality experiments: INKLING_KV_SIM=<mode>.

Modes: off | fp8 | rtn<bits> (uniform, per-vector absmax scale) | tq<bits> (TurboQuant-style: fixed random-sign
Hadamard rotation + Lloyd-Max codebook for a unit-variance Gaussian, per-vector RMS scale) | k<kb>v<vb> mixes
(e.g. k8v4: K as rtn8, V as tq4). Only for measuring quality; no memory/speed effect.
"""
from __future__ import annotations

import functools
import math
import os

import torch

MODE = [os.environ.get("INKLING_KV_SIM", "off")]


def set_mode(m: str) -> None:
    MODE[0] = m


@functools.lru_cache(maxsize=None)
def _codebook(bits: int) -> torch.Tensor:
    """Lloyd-Max levels for N(0,1) (computed numerically)."""
    n = 1 << bits
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2_000_000, generator=g, dtype=torch.float64)
    lv = torch.linspace(-2.5, 2.5, n, dtype=torch.float64)
    for _ in range(60):
        edges = (lv[1:] + lv[:-1]) / 2
        idx = torch.bucketize(x, edges)
        lv = torch.stack([x[idx == i].mean() if (idx == i).any() else lv[i] for i in range(n)])
    return lv.float()


@functools.lru_cache(maxsize=None)
def _hadamard(d: int, device_str: str) -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < d:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    g = torch.Generator().manual_seed(1234)
    signs = (torch.randint(0, 2, (d,), generator=g) * 2 - 1).float()
    return ((h * signs[None, :]) / math.sqrt(d)).to(device_str)  # orthogonal: R = H diag(signs) / sqrt(d)


def tq_roundtrip(x: torch.Tensor, bits: int) -> torch.Tensor:
    d = x.shape[-1]
    r = _hadamard(d, str(x.device))
    xr = x.float() @ r.T
    scale = xr.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-8)
    lv = _codebook(bits).to(x.device)
    edges = (lv[1:] + lv[:-1]) / 2
    q = lv[torch.bucketize((xr / scale).contiguous(), edges)] * scale
    return (q @ r).to(x.dtype)


def rtn_roundtrip(x: torch.Tensor, bits: int) -> torch.Tensor:
    xf = x.float()
    qmax = (1 << (bits - 1)) - 1
    scale = xf.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
    return (torch.round(xf / scale).clamp(-qmax - 1, qmax) * scale).to(x.dtype)


def _apply(x: torch.Tensor, spec: str) -> torch.Tensor:
    if spec == "off":
        return x
    if spec == "fp8":
        return x.to(torch.float8_e4m3fn).to(x.dtype)
    if spec.startswith("tq"):
        return tq_roundtrip(x, int(spec[2:]))
    if spec.startswith("rtn"):
        return rtn_roundtrip(x, int(spec[3:]))
    raise ValueError(spec)


def simulate(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m = MODE[0]
    if m == "off":
        return k, v
    if m.startswith("k") and "v" in m[1:] and m[1].isdigit():  # e.g. k8v4, k4v4
        kb, vb = m[1:].split("v")
        return _apply(k, "rtn" + kb if int(kb) >= 8 else "tq" + kb), _apply(v, "tq" + vb)
    return _apply(k, m), _apply(v, m)
