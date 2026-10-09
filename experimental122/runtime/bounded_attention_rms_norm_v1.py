"""Inference dependency extracted from validated experimental runtime."""

import hashlib, inspect, types

import torch

from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as q

FORWARD_SHA='028dcc012a97674063c6d9e57f99246d081a9af06307da252fe551caacab1899'

INNER_SHA='80a288dc4656923e12d7f924ef6e3ecc1255fb3d0075cde0c11c95378d7d2d51'

def bounded_attention_rms_forward(self,x,max_rows=4096):
 assert not torch.is_grad_enabled() and x.ndim==4 and x.shape[0]==1 and x.shape[-1]==256 and x.dtype==torch.bfloat16
 if x.shape[1]<=4096:return q.Qwen3_5MoeRMSNorm.forward(self,x)
 flat=x.reshape(-1,x.shape[-1]);assert flat.untyped_storage().data_ptr()==x.untyped_storage().data_ptr(),'Input flatten must not materialize query storage'
 output=torch.empty(x.shape,device=x.device,dtype=x.dtype);rows=output.reshape(-1,x.shape[-1])
 for start in range(0,len(flat),max_rows):
  end=min(start+max_rows,len(flat));rows[start:end].copy_(q.Qwen3_5MoeRMSNorm.forward(self,flat[start:end]))
 return output

def bind_bounded_attention_rms_layer(layer):
 assert hashlib.sha256(inspect.getsource(q.Qwen3_5MoeRMSNorm.forward).strip().encode()).hexdigest()==FORWARD_SHA
 assert hashlib.sha256(inspect.getsource(q.Qwen3_5MoeRMSNorm._norm).strip().encode()).hexdigest()==INNER_SHA
 if hasattr(layer,'self_attn'):
  for norm in (layer.self_attn.q_norm,layer.self_attn.k_norm):
   assert type(norm) is q.Qwen3_5MoeRMSNorm
   norm.forward=types.MethodType(bounded_attention_rms_forward,norm)

def bind_bounded_attention_rms_model(model):
 for layer in model.model.language_model.layers:bind_bounded_attention_rms_layer(layer)
