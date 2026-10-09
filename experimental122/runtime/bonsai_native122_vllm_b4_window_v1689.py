"""Compose the independently validated B4 binding and bounded rotary loader.

No projection body changes. The two parent loaders cooperatively call super:
window -> B4 proof -> original weights, then bounded-cache GPU verification.
Whole-engine B4 eager/graph equality still requires its own numerical control.
"""
from bonsai_native122_vllm_request_prefill_v1662 import (
    Native122BatchedPackedModelLoader, ARCHITECTURE, register_model, MIXED_AUDIT,
)
from bonsai_native122_vllm_text_window_v1656 import (
    Native122TextWindowPackedModelLoader, text_window_overrides as serial_window_overrides,
)
from vllm.model_executor.model_loader import register_model_loader

LOAD_FORMAT = 'bonsai_native122_b4_text_window_packed_v1'


def text_window_overrides(config, artifact):
    config = serial_window_overrides(config, artifact)
    if hasattr(config, 'text_config'):
        config.architectures = [ARCHITECTURE]
    return config


@register_model_loader(LOAD_FORMAT)
class Native122B4TextWindowPackedModelLoader(
    Native122TextWindowPackedModelLoader, Native122BatchedPackedModelLoader,
):
    pass
