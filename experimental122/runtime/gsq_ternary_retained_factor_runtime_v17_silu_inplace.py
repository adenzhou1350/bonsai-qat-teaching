"""Inference dependency extracted from validated experimental runtime."""

from indexed_gate_base_low_projection_v1 import indexed_gate_base_low_projection

from bounded_ternary_base_projection_v2_mm_out import bounded_ternary_base_projection

from bounded_grouped_residual_v1 import bounded_grouped_residual_add

import torch

from bounded_route_reduce_v1 import bounded_route_reduce

from fixed_route_mm_out_v1 import fixed_route_mm_out

from gsq_ternary_retained_factor_runtime_v1 import RetainedFactorTritBank

from gsq_ternary_packed_runtime import PackedGSQTritExperts

class GroupedRetainedFactorTritBank(RetainedFactorTritBank):
 def grouped_projection(self,x,offsets):
  assert not torch.is_grad_enabled(), 'Inference-only buffer reuse'
  from transformers.integrations.moe import _grouped_mm
  base=bounded_ternary_base_projection(self,x,offsets)
  low=_grouped_mm(x,self.lora_a.to(x.dtype).transpose(-2,-1),offsets)
  bounded_grouped_residual_add(base,low,self.lora_b.to(x.dtype).transpose(-2,-1),offsets,self.lora_scale)
  del low
  return base.to(torch.bfloat16)
 def gemv(self,x,ids,per_route=False):
  e,n,k=self.shape;routes=len(ids);assert ids.ndim==1 and x.dtype==torch.bfloat16 and x.numel()==k*(routes if per_route else 1)
  inputs=x.reshape(routes if per_route else 1,k)
  if not per_route:inputs=inputs.expand(routes,k).contiguous()
  assert routes==8
  weights=self.weights(ids);base=fixed_route_mm_out(inputs,weights.transpose(-2,-1));del weights
  low=fixed_route_mm_out(inputs,self.lora_a[ids].to(x.dtype).transpose(-2,-1))
  correction=fixed_route_mm_out(low,self.lora_b[ids].to(x.dtype).transpose(-2,-1))
  return (base+correction*self.lora_scale).to(torch.bfloat16)

class PackedGroupedRetainedFactorTritExperts(PackedGSQTritExperts):
 def __init__(self,banks):
  torch.nn.Module.__init__(self);self.gate=GroupedRetainedFactorTritBank(banks['gate_up_proj']);self.down=GroupedRetainedFactorTritBank(banks['down_proj']);e,n,k=self.gate.shape;assert self.down.shape==[e,k,n//2]

 def forward(self,hidden,selected,probabilities):
  if len(hidden)==1:return super().forward(hidden,selected,probabilities)
  assert not torch.is_grad_enabled()
  x=hidden.to(torch.bfloat16).contiguous();fanout=selected.shape[-1];ids,order=selected.reshape(-1).sort()
  ends=torch.bincount(ids,minlength=self.gate.shape[0]).cumsum(0).to(torch.int32)
  from transformers.integrations.moe import _grouped_mm
  base,low=indexed_gate_base_low_projection(self.gate,x,order,ends,fanout)
  bounded_grouped_residual_add(base,low,self.gate.lora_b.to(base.dtype).transpose(-2,-1),ends,self.gate.lora_scale)
  del low
  gate,up=base.to(torch.bfloat16).chunk(2,-1)
  del base
  middle=torch.nn.functional.silu(gate).mul_(up)
  # Gate/up and routed inputs are dead before the down projection.
  del gate,up
  base=bounded_ternary_base_projection(self.down,middle,ends)
  low=_grouped_mm(middle,self.down.lora_a.to(middle.dtype).transpose(-2,-1),ends)
  del middle
  bounded_grouped_residual_add(base,low,self.down.lora_b.to(base.dtype).transpose(-2,-1),ends,self.down.lora_scale)
  del low
  result=base.to(torch.bfloat16)
  del base
  inverse=torch.empty_like(order);inverse[order]=torch.arange(len(order),device=order.device)
  output=bounded_route_reduce(result,probabilities,inverse)
  return output.to(hidden.dtype)
