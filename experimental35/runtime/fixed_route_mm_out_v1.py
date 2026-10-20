"""Fixed eight-route specialization of Torch2.11 GroupedMMUtils.h fallback.

Uses the same BF16 mm_out operator/output slices. No dynamic offsets D2H copy.
Independent numerical admission is required; this helper provides none.
"""
import torch
def fixed_route_mm_out(inputs,weights):
 assert inputs.ndim==2 and weights.ndim==3 and inputs.shape[0]==weights.shape[0]==8 and inputs.shape[-1]==weights.shape[-2] and inputs.dtype==weights.dtype==torch.bfloat16
 result=torch.empty((8,weights.shape[-1]),device=inputs.device,dtype=inputs.dtype)
 for route in range(8):torch.mm(inputs[route:route+1],weights[route],out=result[route:route+1])
 return result
