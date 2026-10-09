"""Inference dependency extracted from validated experimental runtime."""

import hashlib, inspect, types

import torch

import torch.nn.functional as F

from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as q

ORIGINAL_FORWARD_SHA256='d7a4c707ff14918121a5cc328e32bb962c460880ff7e6ab243000c7c5b396ddd'

def original_norm_forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))

        return hidden_states.to(input_dtype)

def bounded_gated_norm_forward(self,hidden_states,gate=None,max_rows=4096):
 assert not torch.is_grad_enabled() and hidden_states.ndim==2 and gate is not None and gate.shape==hidden_states.shape and hidden_states.dtype==torch.bfloat16
 if hidden_states.shape[0]<=max_rows:return original_norm_forward(self,hidden_states,gate)
 output=torch.empty_like(hidden_states)
 for start in range(0,hidden_states.shape[0],max_rows):
  end=min(start+max_rows,hidden_states.shape[0]);output[start:end]=original_norm_forward(self,hidden_states[start:end],gate[start:end])
 return output

def bind_gated_norm_layer(layer):
 assert hashlib.sha256(inspect.getsource(q.Qwen3_5MoeRMSNormGated.forward).strip().encode()).hexdigest()==ORIGINAL_FORWARD_SHA256
 if hasattr(layer,'linear_attn'):
  norm=layer.linear_attn.norm;assert type(norm) is q.Qwen3_5MoeRMSNormGated
  norm.forward=types.MethodType(bounded_gated_norm_forward,norm)

def bind_gated_norm_model(model):
 for layer in model.model.language_model.layers:bind_gated_norm_layer(layer)
