"""Inference dependency extracted from validated experimental runtime."""

import torch

from bounded_route_reduce_v1 import bounded_route_reduce

from gsq_ternary_retained_factor_runtime_v17_silu_inplace import PackedGroupedRetainedFactorTritExperts as PreviousExperts

class PackedGroupedRetainedFactorTritExperts(PreviousExperts):
 def forward(self,hidden,selected,probabilities):
  if len(hidden)<=4096:return super().forward(hidden,selected,probabilities)
  assert not torch.is_grad_enabled() and hidden.dtype==torch.bfloat16 and probabilities.shape==selected.shape
  x=hidden.contiguous();fanout=selected.shape[-1];assert fanout==8
  ids,order=selected.reshape(-1).sort();ends=torch.bincount(ids,minlength=self.gate.shape[0]).cumsum(0).to(torch.int32);del ids
  stops=ends.cpu().tolist();e,n,k=self.gate.shape;assert e==256 and self.down.shape==[e,k,n//2] and self.gate.lora_scale==self.down.lora_scale==1.
  result=torch.empty((len(order),k),device=x.device,dtype=torch.bfloat16)
  for expert,hi in enumerate(stops):
   lo=stops[expert-1] if expert else 0
   if hi==lo:continue
   routed=x[order[lo:hi]//fanout];count=hi-lo;assert count<=len(x)
   eid=torch.tensor([expert],device=x.device,dtype=torch.int64);weight=self.gate.weights(eid)
   base=torch.empty((count,n),device=x.device,dtype=torch.bfloat16);torch.mm(routed,weight[0].transpose(-2,-1),out=base);del weight
   low=torch.empty((count,8),device=x.device,dtype=torch.bfloat16);torch.mm(routed,self.gate.lora_a[expert].transpose(-2,-1),out=low);del routed
   correction=torch.empty_like(base);torch.mm(low,self.gate.lora_b[expert].transpose(-2,-1),out=correction);correction.mul_(1.);base.add_(correction);del correction,low
   gate,up=base.chunk(2,-1);middle=torch.nn.functional.silu(gate).mul_(up);del gate,up,base
   weight=self.down.weights(eid);torch.mm(middle,weight[0].transpose(-2,-1),out=result[lo:hi]);del weight,eid
   low=torch.empty((count,8),device=x.device,dtype=torch.bfloat16);torch.mm(middle,self.down.lora_a[expert].transpose(-2,-1),out=low);del middle
   correction=torch.empty((count,k),device=x.device,dtype=torch.bfloat16);torch.mm(low,self.down.lora_b[expert].transpose(-2,-1),out=correction);correction.mul_(1.);result[lo:hi].add_(correction);del correction,low
  inverse=torch.empty_like(order);inverse[order]=torch.arange(len(order),device=order.device);output=bounded_route_reduce(result,probabilities,inverse)
  return output.to(hidden.dtype)
