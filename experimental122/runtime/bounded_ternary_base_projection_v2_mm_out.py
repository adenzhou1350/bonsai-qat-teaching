"""Inference dependency extracted from validated experimental runtime."""

import torch

def bounded_ternary_base_projection(self,x,offsets):
 assert not torch.is_grad_enabled() and x.dtype==torch.bfloat16
 e,n,k=self.shape;ends=offsets.detach().cpu().tolist();assert e==256 and ends[-1]==len(x) and x.shape[1]==k
 result=torch.empty((len(x),n),device=x.device,dtype=torch.bfloat16)
 for expert in range(e):
  lo=ends[expert-1] if expert else 0;hi=ends[expert]
  if hi==lo:continue
  weight=self.weights(torch.tensor([expert],device=x.device,dtype=torch.int64))
  assert weight.shape==(1,n,k) and weight.dtype==x.dtype
  torch.mm(x[lo:hi],weight[0].transpose(-2,-1),out=result[lo:hi]);del weight
 return result
