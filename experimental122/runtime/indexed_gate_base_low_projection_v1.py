"""Inference dependency extracted from validated experimental runtime."""

import torch

def indexed_gate_base_low_projection(bank,x,order,ends,fanout):
 assert not torch.is_grad_enabled() and x.dtype==torch.bfloat16 and order.dtype==torch.int64
 e,n,k=bank.shape;stops=ends.detach().cpu().tolist();assert e==256 and x.shape[1]==k and stops[-1]==len(order) and bank.lora_a.shape==(e,8,k)
 base=torch.empty((len(order),n),device=x.device,dtype=torch.bfloat16);low=torch.empty((len(order),8),device=x.device,dtype=torch.bfloat16)
 for expert,hi in enumerate(stops):
  lo=stops[expert-1] if expert else 0
  if hi==lo:continue
  routed=x[order[lo:hi]//fanout]
  weight=bank.weights(torch.tensor([expert],device=x.device,dtype=torch.int64))
  torch.mm(routed,weight[0].transpose(-2,-1),out=base[lo:hi]);del weight
  torch.mm(routed,bank.lora_a[expert].to(x.dtype).transpose(-2,-1),out=low[lo:hi]);del routed
 return base,low
