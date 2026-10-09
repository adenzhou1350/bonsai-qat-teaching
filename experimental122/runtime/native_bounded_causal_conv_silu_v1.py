"""Inference dependency extracted from validated experimental runtime."""

import torch

def bounded_causal_conv_silu(module,x,channels_per_chunk=256):
 if x.shape[-1]<=4096:return torch.nn.functional.silu(module(x)[:,:,:x.shape[-1]])
 assert not torch.is_grad_enabled() and x.dtype==module.weight.dtype==torch.bfloat16
 assert isinstance(module,torch.nn.Conv1d) and module.in_channels==module.out_channels==module.groups==x.shape[1]
 assert module.kernel_size==(4,) and module.padding==(3,) and module.stride==module.dilation==(1,) and module.padding_mode=='zeros'
 assert not module._forward_hooks and not module._forward_pre_hooks
 result=torch.empty(x.shape,device=x.device,dtype=x.dtype)
 for begin in range(0,x.shape[1],channels_per_chunk):
  end=min(begin+channels_per_chunk,x.shape[1]);part=x[:,begin:end].contiguous()
  y=torch.nn.functional.conv1d(part,module.weight[begin:end],None if module.bias is None else module.bias[begin:end],stride=1,padding=3,dilation=1,groups=end-begin)
  activated=torch.nn.functional.silu(y[:,:,:x.shape[-1]])
  result[:,begin:end].copy_(activated);del part,y,activated
 return result
