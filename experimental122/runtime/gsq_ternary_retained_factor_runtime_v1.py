"""Inference dependency extracted from validated experimental runtime."""

import torch

from gsq_ternary_linear_runtime_v2 import LinearTritBank

class RetainedFactorTritBank(LinearTritBank):
    def __init__(self,bank):
        super().__init__(bank);e,n,k=self.shape;factor=bank['retained_factor']
        assert factor['format']=='retained_rank8_bf16_factors_v1' and factor['rank']==8
        assert factor['a'].shape==(e,8,k) and factor['b'].shape==(e,n,8)
        assert factor['a'].dtype==factor['b'].dtype==torch.bfloat16
        assert factor['scale']==1.0
        self.register_buffer('lora_a',factor['a'].to('cuda'));self.register_buffer('lora_b',factor['b'].to('cuda'));self.lora_scale=1.0
    def grouped_projection(self,x,offsets):
        from transformers.integrations.moe import _grouped_mm
        base=super().grouped_projection(x,offsets)
        low=_grouped_mm(x,self.lora_a.to(x.dtype).transpose(-2,-1),offsets)
        correction=_grouped_mm(low,self.lora_b.to(x.dtype).transpose(-2,-1),offsets)
        return (base+correction*self.lora_scale).to(torch.bfloat16)
    def gemv(self,x,ids,per_route=False):
        base=super().gemv(x,ids,per_route);e,n,k=self.shape
        x=x.reshape(len(ids) if per_route else 1,k)
        a=self.lora_a[ids].to(x.dtype);b=self.lora_b[ids].to(x.dtype)
        correction=torch.empty_like(base)
        for route in range(len(ids)):
            low=torch.nn.functional.linear(x[route:route+1] if per_route else x,a[route])
            correction[route:route+1]=torch.nn.functional.linear(low,b[route])
        return (base+correction*self.lora_scale).to(torch.bfloat16)
