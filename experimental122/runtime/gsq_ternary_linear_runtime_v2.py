"""Inference dependency extracted from validated experimental runtime."""

import torch

from gsq_ternary_packed_runtime import TritBank

class LinearTritBank(TritBank):
 def gemv(self,x,ids,per_route=False):
  e,n,k=self.shape;assert ids.ndim==1 and ids.is_cuda
  x=x.reshape(len(ids) if per_route else 1,k);weights=self.weights(ids)
  result=torch.empty((len(ids),n),device=x.device,dtype=torch.bfloat16)
  for route in range(len(ids)):
   result[route:route+1]=torch.nn.functional.linear(x[route:route+1] if per_route else x,weights[route])
  return result
