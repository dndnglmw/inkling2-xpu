# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2: platform-specific implementations live under `nvidia/` and `xpu/`;
configs, processing, quantization metadata and the DSpark speculator are
shared."""

from typing import TYPE_CHECKING

from vllm.platforms import current_platform

if TYPE_CHECKING:
    from .nvidia.dspark import Inkling2DSparkModel
    from .nvidia.loop import Inkling2LoopForConditionalGeneration
    from .nvidia.model import Inkling2ForConditionalGeneration

__all__ = [
    "Inkling2DSparkModel",
    "Inkling2ForConditionalGeneration",
    "Inkling2LoopForConditionalGeneration",
]


def __getattr__(name: str):
    if name in (
        "Inkling2ForConditionalGeneration",
        "Inkling2LoopForConditionalGeneration",
    ):
        if current_platform.is_xpu():
            from .xpu import model as xpu_model

            return xpu_model.Inkling2XPUForConditionalGeneration
        if name == "Inkling2LoopForConditionalGeneration":
            from .nvidia import loop

            return loop.Inkling2LoopForConditionalGeneration
        from .nvidia import model

        return model.Inkling2ForConditionalGeneration
    if name == "Inkling2DSparkModel":
        if current_platform.is_xpu():
            from .xpu import dspark as xpu_dspark

            return xpu_dspark.Inkling2XPUDSparkModel
        from .nvidia import dspark

        return dspark.Inkling2DSparkModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
