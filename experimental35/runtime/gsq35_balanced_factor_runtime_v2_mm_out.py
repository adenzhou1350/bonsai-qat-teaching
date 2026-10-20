"""GSQ INT2 frozen bases with separately retained BF16 factors."""
import torch,triton
import triton.language as tl
from gemq_packed_runtime import _weight,PackedGEMQExperts
from gemq_packed_runtime_chunked import PackedGEMQChunkedBank

@triton.jit
def _selected(C,O,B,S,Z,Slots,Fallback,Ids,Out,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
 row=tl.program_id(0)*BN+tl.arange(0,BN);col=tl.program_id(1)*BK+tl.arange(0,BK);route=tl.program_id(2);expert=tl.load(Ids+route)
 weight=_weight(C,O,B,S,Z,Slots,Fallback,expert,row,col,N,K)
 tl.store(Out+(route*N+row[:,None])*K+col[None,:],weight,(row[:,None]<N)&(col[None,:]<K))

class GSQFactorBank(PackedGEMQChunkedBank):
 def gemv(self,x,ids,per_route=False):
  from fixed_route_mm_out_v1 import fixed_route_mm_out
  _,n,k=self.shape;routes=len(ids);assert x.is_cuda and ids.is_cuda and x.dtype==torch.bfloat16 and x.numel()==k*(routes if per_route else 1)
  weight=torch.empty((routes,n,k),device=x.device,dtype=torch.bfloat16)
  _selected[(triton.cdiv(n,4),triton.cdiv(k,256),routes)](*self.pointers(),ids.contiguous(),weight,n,k,4,256,num_warps=4,enable_fp_fusion=False)
  routed=x.reshape(routes if per_route else 1,k)
  if not per_route:routed=routed.expand(routes,k).contiguous()
  assert routes==8
  output=fixed_route_mm_out(routed,weight.transpose(-2,-1))
  low=fixed_route_mm_out(routed,self.lora_a[ids].transpose(-2,-1))
  return (output+fixed_route_mm_out(low,self.lora_b[ids].transpose(-2,-1))*.125).to(torch.bfloat16)

class PackedGSQFactorExperts(PackedGEMQExperts):
 def __init__(self,banks):
  torch.nn.Module.__init__(self);self.gate=GSQFactorBank(banks['gate_up_proj']);self.down=GSQFactorBank(banks['down_proj']);e,n,k=self.gate.shape;assert self.down.shape==[e,k,n//2]
 def forward(self,hidden,selected,probabilities):
  if len(hidden)>1:return super().forward(hidden,selected,probabilities)
  x=hidden.to(torch.bfloat16).contiguous();ids,order=selected[0].sort();gate,up=self.gate.gemv(x,ids).chunk(2,-1);middle=(torch.nn.functional.silu(gate)*up).contiguous();output=self.down.gemv(middle,ids,per_route=True);weighted=output*probabilities[0,order,None];inverse=torch.empty_like(order);inverse[order]=torch.arange(len(order),device=order.device)
  return weighted[inverse].sum(0,keepdim=True).to(hidden.dtype)

def unpack_original_base(state):
 """Independent CPU bit decoding, using neither Triton nor packed runtime."""
 import numpy as np
 assert state['format']=='own_contiguous_bit_cpu_v1' and state['nbits']==2 and state['group_size']==128 and state['source_dtype']=='torch.bfloat16'
 n,k=state['shape'];bits=np.unpackbits(state['codes'].numpy(),bitorder='little');assert len(bits)==n*k*2;codes=torch.from_numpy((bits.reshape(-1,2)*np.array([1,2])).sum(1).astype(np.uint8)).to(torch.bfloat16).reshape(-1,128)
 return ((codes-state['zeros'])*state['scales']).reshape(n,k)

class HardGSQFactorExperts(torch.nn.Module):
 """Independent unpacked base and separate factors; no merged weight rounding."""
 def __init__(self,banks):
  super().__init__();self.e,self.n,self.k=banks['gate_up_proj']['shape']
  for name,bank in banks.items():
   self.register_buffer(name,torch.stack([unpack_original_base(s) for s in bank['states']]).to('cuda'))
   self.register_buffer(name+'_A',torch.stack([s['lora_A'] for s in bank['states']]).to('cuda'));self.register_buffer(name+'_B',torch.stack([s['lora_B'] for s in bank['states']]).to('cuda'))
 def projection(self,x,ends,name):
  from transformers.integrations.moe import _grouped_mm
  host=ends.detach().cpu().tolist();parts=[]
  for start in range(0,self.e,16):
   stop=min(start+16,self.e);begin=host[start-1] if start else 0;end=host[stop-1]
   if begin==end:continue
   inputs=x[begin:end];offsets=(ends[start:stop]-begin).contiguous();base=getattr(self,name)[start:stop];A=getattr(self,name+'_A')[start:stop];B=getattr(self,name+'_B')[start:stop];y=_grouped_mm(inputs,base.transpose(-2,-1),offsets);low=_grouped_mm(inputs,A.transpose(-2,-1),offsets);parts.append((y+_grouped_mm(low,B.transpose(-2,-1),offsets)*.125).to(torch.bfloat16))
  return torch.cat(parts)
 def forward(self,hidden,selected,probabilities):
  ids,order=selected.reshape(-1).sort();ends=torch.bincount(ids,minlength=self.e).cumsum(0).to(torch.int32);inputs=hidden[order//selected.shape[-1]];gate,up=self.projection(inputs,ends,'gate_up_proj').chunk(2,-1);middle=torch.nn.functional.silu(gate)*up;output=self.projection(middle,ends,'down_proj');weighted=output*probabilities.reshape(-1)[order,None];inverse=torch.empty_like(order);inverse[order]=torch.arange(len(order),device=order.device)
  return weighted[inverse].reshape(len(hidden),selected.shape[-1],-1).sum(1).to(hidden.dtype)
