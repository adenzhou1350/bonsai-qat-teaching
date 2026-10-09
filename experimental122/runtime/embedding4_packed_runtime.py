"""Inference dependency extracted from validated experimental runtime."""

import torch, triton

import triton.language as tl

@triton.jit
def _gather(Q,S,Z,IDS,Y,M:tl.constexpr,K:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0);col=tl.program_id(1)*BK+tl.arange(0,BK)
    token=tl.load(IDS+row);valid=col<K
    byte=tl.load(Q+token*(K//2)+col//2,valid,other=0).to(tl.int32)
    code=((byte>>(4*(col%2)))&15).to(tl.bfloat16)
    offset=token*(K//128)+col//128
    scale=tl.load(S+offset,valid,other=0);zero=tl.load(Z+offset,valid,other=0)
    value=((code-zero).to(tl.bfloat16)*scale).to(tl.bfloat16)
    tl.store(Y+row*K+col,value,valid)

class PackedAffine4Embedding(torch.nn.Module):
    def __init__(self,state):
        super().__init__()
        assert state['format']=='own_affine_dense4_v1' and state['role']=='embedding' and state['group_size']==128 and state['source_dtype']=='torch.bfloat16'
        self.num_embeddings,self.embedding_dim=state['shape'];n,k=state['shape']
        assert k%128==0 and state['codes'].shape==(n,k//2) and state['codes'].dtype==torch.uint8
        assert state['scales'].shape==state['zeros'].shape==(n,k//128) and state['scales'].dtype==state['zeros'].dtype==torch.bfloat16
        rank=state['embedding_A'].shape[1]
        assert state['embedding_A'].shape==(n,rank) and state['embedding_B'].shape==(k,rank) and state['embedding_A'].dtype==state['embedding_B'].dtype==torch.bfloat16
        for name in ('codes','scales','zeros','embedding_A','embedding_B'):self.register_buffer(name,state[name].to('cuda').contiguous())
        self.lora_scale=state['lora_scale'];assert self.lora_scale==.125
    def base(self,ids):
        assert ids.is_cuda and ids.dtype==torch.int64
        ids=ids.contiguous();m=ids.numel();k=self.embedding_dim
        y=torch.empty((m,k),device=ids.device,dtype=torch.bfloat16)
        _gather[(m,triton.cdiv(k,256))](self.codes,self.scales,self.zeros,ids,y,m,k,256,num_warps=4,enable_fp_fusion=False)
        return y.reshape(*ids.shape,k)
    def forward(self,ids):
        base=self.base(ids)
        correction=torch.nn.functional.linear(self.embedding_A[ids],self.embedding_B)*self.lora_scale
        return base+correction
