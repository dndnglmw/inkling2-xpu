# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Intel XPU implementation of Inkling 2 Mini (looped)."""

from .model import (
    Inkling2XPUForConditionalGeneration,
    Inkling2XPULoopModel,
    Inkling2XPUModel,
)

__all__ = [
    "Inkling2XPUForConditionalGeneration",
    "Inkling2XPULoopModel",
    "Inkling2XPUModel",
]
