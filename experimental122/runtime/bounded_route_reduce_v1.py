"""Inference dependency extracted from validated experimental runtime."""

import torch

def bounded_route_reduce(result,probabilities,inverse,max_tokens=128):
 assert not torch.is_grad_enabled() and max_tokens>0
 tokens,fanout=probabilities.shape;assert result.ndim==2 and result.shape[0]==tokens*fanout and inverse.shape==(tokens*fanout,) and fanout==8
 dtype=torch.result_type(result,probabilities);out=torch.empty((tokens,result.shape[-1]),device=result.device,dtype=dtype)
 flat=probabilities.reshape(-1)
 for start in range(0,tokens,max_tokens):
  end=min(start+max_tokens,tokens);rows=slice(start*fanout,end*fanout)
  # Gather before pointwise multiplication, retaining each route's same operands.
  gathered=result[inverse[rows]];weighted=gathered*flat[rows,None];del gathered
  out[start:end]=weighted.reshape(end-start,fanout,-1).sum(1)
  del weighted
 return out
