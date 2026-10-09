"""Inference dependency extracted from validated experimental runtime."""

import hashlib

from pathlib import Path

import torch

from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as q

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while part:=f.read(8*1024*1024):h.update(part)
    return h.hexdigest()

def backend():
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    q.FusedRMSNormGated=None;q.causal_conv1d_fn=None;q.chunk_gated_delta_rule=None;q.fused_recurrent_gated_delta_rule=None
