"""Inference dependency extracted from validated experimental runtime."""

import torch, triton

import triton.language as tl

@triton.jit
def _decode(Q,S,Z,W,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0)*BN+tl.arange(0,BN);col=tl.program_id(1)*BK+tl.arange(0,BK);valid=(row[:,None]<N)&(col[None,:]<K)
    byte=tl.load(Q+row[:,None]*(K//2)+col[None,:]//2,valid,other=0).to(tl.int32);code=((byte>>(4*(col[None,:]%2)))&15).to(tl.bfloat16);offset=row[:,None]*(K//128)+col[None,:]//128;scale=tl.load(S+offset,valid,other=0);zero=tl.load(Z+offset,valid,other=0);delta=(code-zero).to(tl.bfloat16);weight=(delta*scale).to(tl.bfloat16);tl.store(W+row[:,None]*K+col[None,:],weight,valid)

@triton.jit
def _gemv(Q,S,Z,X,BIAS,Y,N:tl.constexpr,K:tl.constexpr,HAS_BIAS:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0)*BN+tl.arange(0,BN);cols=tl.arange(0,BK);acc=tl.full((BN,),0,tl.float32)
    for start in range(tl.cdiv(K,BK)):
        col=start*BK+cols;valid=(row[:,None]<N)&(col[None,:]<K);byte=tl.load(Q+row[:,None]*(K//2)+col[None,:]//2,valid,other=0).to(tl.int32);code=((byte>>(4*(col[None,:]%2)))&15).to(tl.bfloat16);offset=row[:,None]*(K//128)+col[None,:]//128;scale=tl.load(S+offset,valid,other=0);zero=tl.load(Z+offset,valid,other=0);weight=((code-zero).to(tl.bfloat16)*scale).to(tl.bfloat16).to(tl.float32);x=tl.load(X+col,col<K,other=0).to(tl.float32);acc+=tl.sum(weight*x[None,:],1)
    if HAS_BIAS:acc+=tl.load(BIAS+row,row<N,other=0).to(tl.float32)
    tl.store(Y+row,acc,row<N)

class PackedAffine4Linear(torch.nn.Module):
    def __init__(self,state,bias=None):
        super().__init__();assert state['format']=='own_affine_dense4_v1' and state['group_size']==128 and state['source_dtype']=='torch.bfloat16';self.out_features,self.in_features=state['shape'];n,k=state['shape'];assert k%128==0 and state['codes'].shape==(n,k//2) and state['codes'].dtype==torch.uint8 and state['scales'].shape==state['zeros'].shape==(n,k//128) and state['scales'].dtype==state['zeros'].dtype==torch.bfloat16
        for name in ('codes','scales','zeros'):self.register_buffer(name,state[name].to('cuda').contiguous())
        self.has_bias=bias is not None;self.register_buffer('bias',bias.detach().to('cuda').contiguous() if bias is not None else torch.empty(0,device='cuda',dtype=torch.bfloat16))
    def dequant(self):
        n,k=self.out_features,self.in_features;weight=torch.empty((n,k),device='cuda',dtype=torch.bfloat16);_decode[(triton.cdiv(n,4),triton.cdiv(k,256))](self.codes,self.scales,self.zeros,weight,n,k,4,256,enable_fp_fusion=False,num_warps=4);return weight
    def forward(self,x):
        shape=x.shape[:-1];x=x.to(torch.bfloat16).reshape(-1,self.in_features).contiguous()
        if len(x)>1:return torch.nn.functional.linear(x,self.dequant(),self.bias if self.has_bias else None).reshape(*shape,self.out_features)
        y=torch.empty((1,self.out_features),device='cuda',dtype=torch.bfloat16);_gemv[(triton.cdiv(self.out_features,4),)](self.codes,self.scales,self.zeros,x,self.bias,y,self.out_features,self.in_features,self.has_bias,4,512,enable_fp_fusion=False,num_warps=4);return y.reshape(*shape,self.out_features)
