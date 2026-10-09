"""Exact five-trit weight decode using a 512-byte table; no projection changes."""
import hashlib
from pathlib import Path
import torch,triton
import triton.language as tl

def make_lookup():
 values=[sum(((byte//(3**digit))%3)<<(2*digit) for digit in range(5)) for byte in range(256)]
 assert all(((values[byte]>>(2*digit))&3)-1==(byte//(3**digit))%3-1 for byte in range(256) for digit in range(5))
 return torch.tensor(values,device='cuda',dtype=torch.int16)

@triton.jit
def decode_kernel(Q,S,Ids,Lut,Out,N:tl.constexpr,K:tl.constexpr,P:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
 row=tl.program_id(0)*BN+tl.arange(0,BN);col=tl.program_id(1)*BK+tl.arange(0,BK);route=tl.program_id(2);expert=tl.load(Ids+route)
 mask=(row[:,None]<N)&(col[None,:]<K)
 byte=tl.load(Q+(expert*N+row[:,None])*P+col[None,:]//5,mask,other=121).to(tl.int32)
 expanded=tl.load(Lut+byte).to(tl.int32)
 code=((expanded>>(2*(col[None,:]%5)))&3)-1
 scale=tl.load(S+(expert*N+row[:,None])*(K//128)+col[None,:]//128,mask,other=0).to(tl.float32)
 weight=(code.to(tl.float32)*scale).to(tl.bfloat16)
 tl.store(Out+(route*N+row[:,None])*K+col[None,:],weight,mask)

def weights(bank,ids,lookup):
 _,n,k=bank.shape
 assert ids.ndim==1 and ids.dtype==torch.int64 and lookup.shape==(256,) and lookup.dtype==torch.int16
 output=torch.empty((len(ids),n,k),device='cuda',dtype=torch.bfloat16)
 decode_kernel[(triton.cdiv(n,4),triton.cdiv(k,256),len(ids))](bank.codes,bank.scales,ids.contiguous(),lookup,output,n,k,triton.cdiv(k,5),4,256,num_warps=4,enable_fp_fusion=False)
 return output

def install():
 import gsq_ternary_packed_runtime as module
 assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()=='000219642e51d87168ab8f50891f9b0cbde93143cb4c5b3530abe086b599955f'
 original=module.TritBank.weights;lookup=make_lookup()
 def replacement(self,ids):return weights(self,ids,lookup)
 module.TritBank.weights=replacement
 return module,original,lookup
