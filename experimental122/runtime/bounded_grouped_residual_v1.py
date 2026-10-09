"""Inference dependency extracted from validated experimental runtime."""

import torch

def bounded_grouped_residual_add(base,low,weight,offsets,scale):
 assert not torch.is_grad_enabled()
 assert base.ndim==low.ndim==2 and weight.ndim==3
 assert base.dtype==low.dtype==weight.dtype==torch.bfloat16 and scale==1.0
 assert base.device==low.device==weight.device==offsets.device
 assert base.shape==(low.shape[0],weight.shape[-1]) and low.shape[-1]==weight.shape[-2]
 assert base.stride()==(base.shape[-1],1) and base.shape[-1]%8==0
 assert offsets.ndim==1 and offsets.dtype==torch.int32 and len(offsets)==len(weight)
 ends=offsets.cpu().tolist();start=0
 for expert,end in enumerate(ends):
  assert start<=end<=len(low)
  correction=torch.empty((end-start,base.shape[-1]),device=base.device,dtype=base.dtype)
  torch.mm(low[start:end],weight[expert],out=correction)
  correction.mul_(scale)
  base[start:end].add_(correction)
  del correction
  start=end
 assert start==len(low)
 return base
