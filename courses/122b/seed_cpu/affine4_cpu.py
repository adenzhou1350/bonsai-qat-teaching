"""Own unsigned-nibble affine4 storage; CPU decode matches BF16 GPTQ arithmetic."""
import torch

def pack(codes,scales,zeros,shape):
    n,k=shape;assert k%128==0 and codes.numel()==n*k and codes.device.type=='cpu' and torch.equal(codes,codes.round()) and int(codes.min())>=0 and int(codes.max())<=15
    values=codes.to(torch.uint8).reshape(n,k);raw=values[:,::2]|(values[:,1::2]<<4)
    state={'format':'own_affine_dense4_v1','shape':[n,k],'group_size':128,'codes':raw.contiguous(),'scales':scales.reshape(n,k//128).contiguous(),'zeros':zeros.reshape(n,k//128).contiguous(),'source_dtype':'torch.bfloat16'}
    assert torch.equal(decode_codes(state),values);assert state['scales'].dtype==state['zeros'].dtype==torch.bfloat16;return state

def decode_codes(state):
    assert state['format']=='own_affine_dense4_v1' and state['group_size']==128 and state['source_dtype']=='torch.bfloat16';n,k=state['shape'];assert k%128==0 and state['codes'].shape==(n,k//2) and state['codes'].dtype==torch.uint8
    byte=state['codes'];return torch.stack((byte&15,byte>>4),dim=-1).reshape(n,k)

def unpack(state):
    n,k=state['shape'];assert state['scales'].shape==state['zeros'].shape==(n,k//128) and state['scales'].dtype==state['zeros'].dtype==torch.bfloat16
    codes=decode_codes(state).to(torch.bfloat16).reshape(n,k//128,128);return ((codes-state['zeros'][...,None])*state['scales'][...,None]).reshape(n,k)

class CPUAffine4Linear(torch.nn.Module):
    def __init__(self,state,bias=None):
        super().__init__();self.register_buffer('weight',unpack(state));self.register_buffer('bias',bias);self.in_features=state['shape'][1];self.out_features=state['shape'][0]
    def forward(self,x):return torch.nn.functional.linear(x,self.weight,self.bias)
