import hashlib
from pathlib import Path
import torch
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as q
def sha(file):
 h=hashlib.sha256()
 with Path(file).open('rb') as f:
  while raw:=f.read(8*1024*1024):h.update(raw)
 return h.hexdigest()
def tensor_sha(value):
 h=hashlib.sha256();raw=value.detach().contiguous().view(torch.uint8).reshape(-1)
 for start in range(0,len(raw),16*1024*1024):h.update(raw[start:start+16*1024*1024].cpu().numpy().tobytes())
 return h.hexdigest()
def backend():
 torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 q.FusedRMSNormGated=None;q.causal_conv1d_fn=None;q.chunk_gated_delta_rule=None;q.fused_recurrent_gated_delta_rule=None
