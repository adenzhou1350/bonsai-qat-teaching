"""Inference dependency extracted from validated experimental runtime."""

import torch, triton

import triton.language as tl

@triton.jit
def _decode(Q,S,Ids,Out,N:tl.constexpr,K:tl.constexpr,P:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
 row=tl.program_id(0)*BN+tl.arange(0,BN);col=tl.program_id(1)*BK+tl.arange(0,BK);route=tl.program_id(2);expert=tl.load(Ids+route)
 byte=tl.load(Q+(expert*N+row[:,None])*P+col[None,:]//5,(row[:,None]<N)&(col[None,:]<K),other=121).to(tl.int32)
 digit=col%5;power=tl.where(digit==0,1,tl.where(digit==1,3,tl.where(digit==2,9,tl.where(digit==3,27,81))))
 code=byte//power[None,:]%3-1
 scale=tl.load(S+(expert*N+row[:,None])*(K//128)+col[None,:]//128,(row[:,None]<N)&(col[None,:]<K),other=0).to(tl.float32)
 weight=(code.to(tl.float32)*scale).to(tl.bfloat16)
 tl.store(Out+(route*N+row[:,None])*K+col[None,:],weight,(row[:,None]<N)&(col[None,:]<K))

class TritBank(torch.nn.Module):
 def __init__(self,bank):
  super().__init__();assert bank['format']=='own_gsq_five_trits_expert_bank_v1';self.shape=bank['shape'];e,n,k=self.shape
  assert e==256 and k%128==0 and len(bank['states'])==e
  for v in bank['states']:assert v['format']=='own_gsq_five_trits_per_byte_v1' and v['exact_source_bf16_grid'] and v['scales'].dtype==torch.bfloat16
  self.register_buffer('codes',torch.stack([v['codes'] for v in bank['states']]).to('cuda'));self.register_buffer('scales',torch.stack([v['scales'] for v in bank['states']]).to('cuda'));assert self.codes.shape==tuple([e,n,(k+4)//5])
 def weights(self,ids):
  e,n,k=self.shape;weight=torch.empty((len(ids),n,k),device='cuda',dtype=torch.bfloat16)
  _decode[(triton.cdiv(n,4),triton.cdiv(k,256),len(ids))](self.codes,self.scales,ids.contiguous(),weight,n,k,(k+4)//5,4,256,num_warps=4,enable_fp_fusion=False);return weight
 def gemv(self,x,ids,per_route=False):
  from transformers.integrations.moe import _grouped_mm
  e,n,k=self.shape;x=x.reshape(len(ids) if per_route else 1,k)
  if not per_route:x=x.expand(len(ids),k).contiguous()
  weights=self.weights(ids);result=_grouped_mm(x,weights.transpose(-2,-1),torch.arange(1,len(ids)+1,device='cuda',dtype=torch.int32));return result
 def grouped_projection(self,x,offsets):
  from transformers.integrations.moe import _grouped_mm
  e,n,k=self.shape;ends=offsets.detach().cpu().tolist();assert ends[-1]==len(x);result=torch.empty((len(x),n),device='cuda',dtype=torch.bfloat16)
  for begin in range(0,e,16):
   end=min(begin+16,e);lo=ends[begin-1] if begin else 0;hi=ends[end-1]
   if hi==lo:continue
   weight=self.weights(torch.arange(begin,end,device='cuda'));result[lo:hi]=_grouped_mm(x[lo:hi],weight.transpose(-2,-1),(offsets[begin:end]-lo).contiguous());del weight
  return result

class PackedGSQTritExperts(torch.nn.Module):
 def __init__(self,banks):
  super().__init__();self.gate=TritBank(banks['gate_up_proj']);self.down=TritBank(banks['down_proj']);e,n,k=self.gate.shape;assert self.down.shape==[e,k,n//2]
 def forward(self,hidden,selected,probabilities):
  x=hidden.to(torch.bfloat16).contiguous();fanout=selected.shape[-1];ids,order=selected.reshape(-1).sort()
  if len(x)==1:
   gate,up=self.gate.gemv(x,ids).chunk(2,-1);middle=(torch.nn.functional.silu(gate)*up).contiguous();result=self.down.gemv(middle,ids,per_route=True)
  else:
   ends=torch.bincount(ids,minlength=self.gate.shape[0]).cumsum(0).to(torch.int32);routed=x[order//fanout];gate,up=self.gate.grouped_projection(routed,ends).chunk(2,-1);middle=(torch.nn.functional.silu(gate)*up).contiguous();result=self.down.grouped_projection(middle,ends)
  weighted=result*probabilities.reshape(-1)[order,None];inverse=torch.empty_like(order);inverse[order]=torch.arange(len(order),device=order.device);return weighted[inverse].reshape(len(hidden),fanout,-1).sum(1).to(hidden.dtype)
