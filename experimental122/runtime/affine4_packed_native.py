"""Inference dependency extracted from validated experimental runtime."""

import torch

from affine4_packed_runtime import PackedAffine4Linear

class PackedAffine4NativeLinear(PackedAffine4Linear):
    def forward(self,x):
        shape=x.shape[:-1];inputs=x.to(torch.bfloat16).reshape(-1,self.in_features).contiguous()
        return torch.nn.functional.linear(inputs,self.dequant(),self.bias if self.has_bias else None).reshape(*shape,self.out_features)
