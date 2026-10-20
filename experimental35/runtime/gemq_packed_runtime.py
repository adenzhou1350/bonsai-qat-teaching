"""Own Qwen3.5 fused-expert CUDA runtime for contiguous1/2/3bit GPTQ bytes.

Official GEMQ quantizer output; own storage/kernel, explicit prefill temporaries.
"""
import torch,triton
import triton.language as tl
from torch import nn

def stack(bank):
    e,n,k=bank['shape'];assert len(bank['states'])==e and k%128==0;codes=[];offsets=[0];widths=[];scales=[];zeros=[];slots=[];fallback=[];rank=None
    for state in bank['states']:
        if state['format']=='own_contiguous_bit_cpu_v1':
            assert state['shape']==[n,k] and state['group_size']==128 and state['source_dtype']=='torch.bfloat16' and state['nbits'] in (1,2,3)
            assert state['codes'].dtype==torch.uint8 and state['codes'].numel()==(n*k*state['nbits']+7)//8 and state['scales'].shape==state['zeros'].shape==(n*k//128,1)
            assert state['scales'].dtype==state['zeros'].dtype==torch.bfloat16 and torch.isfinite(state['scales']).all() and torch.isfinite(state['zeros']).all()
            raw=state['codes'].reshape(-1);widths.append(state['nbits']);scales.append(state['scales'].reshape(n,k//128));zeros.append(state['zeros'].reshape(n,k//128));slots.append(-1)
        else:
            assert state['format']=='unseen_original_bf16' and state['weight'].shape==(n,k) and state['weight'].dtype==torch.bfloat16
            raw=torch.zeros(n*k//8,dtype=torch.uint8);widths.append(1);scales.append(torch.zeros(n,k//128,dtype=torch.bfloat16));zeros.append(torch.zeros_like(scales[-1]));slots.append(len(fallback));fallback.append(state['weight'])
        codes.append(raw);offsets.append(offsets[-1]+raw.numel())
        if 'lora_A' in state:
            a,b=state['lora_A'],state['lora_B'];assert a.dtype==b.dtype==torch.bfloat16 and a.ndim==b.ndim==2 and a.shape[1]==k and b.shape==(n,a.shape[0]) and state['lora_scale']==.125
            if rank is None:rank=a.shape[0]
            assert rank==a.shape[0] and torch.isfinite(a).all() and torch.isfinite(b).all()
    values={'codes':torch.cat(codes),'offsets':torch.tensor(offsets,dtype=torch.int64),'widths':torch.tensor(widths,dtype=torch.int32),'scales':torch.stack(scales),'zeros':torch.stack(zeros),'fallback_slots':torch.tensor(slots,dtype=torch.int32),'fallback_weights':torch.stack(fallback) if fallback else torch.empty(0,n,k,dtype=torch.bfloat16)}
    if rank is not None:
        values['lora_a']=torch.stack([s['lora_A'] if 'lora_A' in s else torch.zeros(rank,k,dtype=torch.bfloat16) for s in bank['states']]);values['lora_b']=torch.stack([s['lora_B'] if 'lora_B' in s else torch.zeros(n,rank,dtype=torch.bfloat16) for s in bank['states']])
    return values

@triton.jit
def _weight(C,O,B,S,Z,Slots,Fallback,expert,row,col,N:tl.constexpr,K:tl.constexpr):
    slot=tl.load(Slots+expert)
    if slot>=0:
        w=tl.load(Fallback+(slot*N+row[:,None])*K+col[None,:],mask=(row[:,None]<N)&(col[None,:]<K),other=0)
    else:
        bits=tl.load(B+expert);start=tl.load(O+expert);end=tl.load(O+expert+1);position=(row[:,None]*K+col[None,:])*bits;byte=start+position//8;shift=position%8;mask=(row[:,None]<N)&(col[None,:]<K)
        lo=tl.load(C+byte,mask=mask,other=0).to(tl.int32);hi=tl.load(C+byte+1,mask=mask&(byte+1<end),other=0).to(tl.int32);code=((lo|(hi<<8))>>shift)&((1<<bits)-1)
        address=(expert*N+row[:,None])*(K//128)+(col//128)[None,:];scale=tl.load(S+address,mask=mask,other=0).to(tl.float32);zero=tl.load(Z+address,mask=mask,other=0).to(tl.float32)
        difference=(code.to(tl.float32)-zero).to(tl.bfloat16).to(tl.float32);w=(difference*scale).to(tl.bfloat16)
    return w.to(tl.bfloat16)

@triton.jit
def _dequant(C,O,B,S,Z,Slots,Fallback,Out,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0)*BN+tl.arange(0,BN);col=tl.program_id(1)*BK+tl.arange(0,BK);expert=tl.program_id(2);weight=_weight(C,O,B,S,Z,Slots,Fallback,expert,row,col,N,K)
    tl.store(Out+(expert*N+row[:,None])*K+col[None,:],weight,mask=(row[:,None]<N)&(col[None,:]<K))

@triton.jit
def _gemv(C,O,B,S,Z,Slots,Fallback,X,Ids,Out,N:tl.constexpr,K:tl.constexpr,PER_ROUTE:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0)*BN+tl.arange(0,BN);route=tl.program_id(1);expert=tl.load(Ids+route);acc=tl.full((BN,),0,tl.float32)
    for start in range(triton.cdiv(K,BK)):
        col=start*BK+tl.arange(0,BK);weight=_weight(C,O,B,S,Z,Slots,Fallback,expert,row,col,N,K);offset=route*K if PER_ROUTE else 0;x=tl.load(X+offset+col,mask=col<K,other=0).to(tl.float32);acc+=tl.sum(weight.to(tl.float32)*x[None,:],axis=1)
    tl.store(Out+route*N+row,acc.to(tl.bfloat16),mask=row<N)

class PackedGEMQBank(nn.Module):
    def __init__(self,bank):
        super().__init__();self.shape=bank['shape']
        for name,value in stack(bank).items():self.register_buffer(name,value.to('cuda'))
    def pointers(self):return (self.codes,self.offsets,self.widths,self.scales,self.zeros,self.fallback_slots,self.fallback_weights)
    def dequant(self):
        e,n,k=self.shape;weight=torch.empty((e,n,k),device='cuda',dtype=torch.bfloat16);_dequant[(triton.cdiv(n,4),triton.cdiv(k,256),e)](*self.pointers(),weight,n,k,4,256,num_warps=4,enable_fp_fusion=False);return weight
    def grouped_projection(self,x,offsets):
        from transformers.integrations.moe import _grouped_mm
        weight=self.dequant();output=_grouped_mm(x,weight.transpose(-2,-1),offsets);del weight
        if hasattr(self,'lora_a'):
            low=_grouped_mm(x,self.lora_a.transpose(-2,-1),offsets);output=(output+_grouped_mm(low,self.lora_b.transpose(-2,-1),offsets)*.125).to(torch.bfloat16)
        return output
    def gemv(self,x,ids,per_route=False):
        e,n,k=self.shape;assert x.is_cuda and ids.is_cuda and x.dtype==torch.bfloat16 and ids.ndim==1 and x.numel()==k*(len(ids) if per_route else 1)
        output=torch.empty((len(ids),n),device='cuda',dtype=torch.bfloat16);_gemv[(triton.cdiv(n,4),len(ids))](*self.pointers(),x.contiguous(),ids.contiguous(),output,n,k,per_route,4,512,num_warps=4,enable_fp_fusion=False)
        if hasattr(self,'lora_a'):
            x=x.reshape(len(ids) if per_route else 1,k);x=x if per_route else x.expand(len(ids),k);low=torch.bmm(self.lora_a[ids],x[...,None]);output=(output+torch.bmm(self.lora_b[ids],low).squeeze(-1)*.125).to(torch.bfloat16)
        return output

class PackedGEMQExperts(nn.Module):
    def __init__(self,banks):
        super().__init__();self.gate=PackedGEMQBank(banks['gate_up_proj']);self.down=PackedGEMQBank(banks['down_proj']);e,n,k=self.gate.shape;assert self.down.shape==[e,k,n//2]
    def forward(self,hidden_states,selected_experts,routing_weights):
        x=hidden_states.to(torch.bfloat16).contiguous();fanout=selected_experts.shape[-1]
        if len(x)>1:
            ids,order=selected_experts.reshape(-1).sort();offsets=torch.histc(ids.int(),bins=self.gate.shape[0],min=0,max=self.gate.shape[0]-1).cumsum(0,dtype=torch.int32);grouped=x[order//fanout];gate,up=self.gate.grouped_projection(grouped,offsets).chunk(2,-1);middle=(torch.nn.functional.silu(gate)*up).contiguous();output=self.down.grouped_projection(middle,offsets);weighted=output*routing_weights.reshape(-1)[order,None];inverse=torch.empty_like(order);inverse[order]=torch.arange(len(order),device=order.device);return weighted[inverse].reshape(len(x),fanout,-1).sum(1).to(hidden_states.dtype)
        ids=selected_experts[0].contiguous();gate,up=self.gate.gemv(x,ids).chunk(2,-1);middle=(torch.nn.functional.silu(gate)*up).contiguous();output=self.down.gemv(middle,ids,per_route=True);return (output*routing_weights[0,:,None]).sum(0,keepdim=True).to(hidden_states.dtype)
