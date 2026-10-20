"""Bounded BF16 expert slices with Torch grouped GEMM arithmetic for prefill.

Only prefill synchronizes routing offsets to CPU. Packed one-token decode is
inherited unchanged; this is not a graph-capturable prefill implementation.
"""
import torch,triton
import triton.language as tl
from gemq_packed_runtime import PackedGEMQBank,PackedGEMQExperts,_weight

@triton.jit
def _range_dequant(C,O,B,S,Z,Slots,Fallback,Out,N:tl.constexpr,K:tl.constexpr,START:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0)*BN+tl.arange(0,BN);col=tl.program_id(1)*BK+tl.arange(0,BK);local=tl.program_id(2)
    weight=_weight(C,O,B,S,Z,Slots,Fallback,local+START,row,col,N,K)
    tl.store(Out+(local*N+row[:,None])*K+col[None,:],weight,(row[:,None]<N)&(col[None,:]<K))

class PackedGEMQChunkedBank(PackedGEMQBank):
    def grouped_projection(self,x,offsets):
        from transformers.integrations.moe import _grouped_mm
        e,n,k=self.shape;assert x.is_cuda and x.dtype==torch.bfloat16 and x.shape[1]==k and offsets.dtype==torch.int32 and len(offsets)==e
        host_ends=offsets.detach().cpu().tolist();assert host_ends[-1]==len(x)
        output=torch.empty((len(x),n),device=x.device,dtype=torch.bfloat16)
        for start in range(0,e,16):
            end=min(start+16,e);begin_row=host_ends[start-1] if start else 0;end_row=host_ends[end-1]
            if begin_row==end_row:continue
            weight=torch.empty((end-start,n,k),device=x.device,dtype=torch.bfloat16)
            _range_dequant[(triton.cdiv(n,4),triton.cdiv(k,256),end-start)](*self.pointers(),weight,n,k,start,4,256,num_warps=4,enable_fp_fusion=False)
            output[begin_row:end_row]=_grouped_mm(x[begin_row:end_row],weight.transpose(-2,-1),(offsets[start:end]-begin_row).contiguous());del weight
        if hasattr(self,'lora_a'):
            low=_grouped_mm(x,self.lora_a.transpose(-2,-1),offsets);output=(output+_grouped_mm(low,self.lora_b.transpose(-2,-1),offsets)*.125).to(torch.bfloat16)
        return output

class PackedGEMQChunkedExperts(PackedGEMQExperts):
    def __init__(self,banks):
        torch.nn.Module.__init__(self);self.gate=PackedGEMQChunkedBank(banks['gate_up_proj']);self.down=PackedGEMQChunkedBank(banks['down_proj']);e,n,k=self.gate.shape;assert self.down.shape==[e,k,n//2]
