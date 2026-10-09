"""Reuse decoded dense weights while preserving single-row decode GEMM shapes."""
import types
import torch,triton
from affine4_packed_native import PackedAffine4NativeLinear
from primary_gsq122_dense4_runtime_v1 import PackedAffine4NativeChunkedHead
from affine4_packed_runtime import _decode
from small_batch122_experts_v2 import bind as bind_selected_routes

def bind(model):
 bind_selected_routes(model)
 for module in model.modules():
  if isinstance(module,PackedAffine4NativeLinear):
   original=module.forward
   def linear(self,x,_original=original):
    count=x.numel()//self.in_features
    if not 2<=count<=4:return _original(x)
    assert not torch.is_grad_enabled()
    shape=x.shape[:-1];inputs=x.to(torch.bfloat16).reshape(-1,self.in_features).contiguous();weight=self.dequant()
    return torch.cat([torch.nn.functional.linear(inputs[i:i+1],weight,self.bias if self.has_bias else None) for i in range(count)],0).reshape(*shape,self.out_features)
   module.forward=types.MethodType(linear,module)
  elif isinstance(module,PackedAffine4NativeChunkedHead):
   original=module.forward
   def head(self,x,_original=original):
    count=x.numel()//self.in_features
    if not 2<=count<=4:return _original(x)
    assert not torch.is_grad_enabled() and not self.has_bias
    shape=x.shape[:-1];inputs=x.to(torch.bfloat16).reshape(-1,self.in_features).contiguous();output=torch.empty((count,self.out_features),device=x.device,dtype=torch.bfloat16)
    for start in range(0,self.out_features,2048):
     stop=min(start+2048,self.out_features);n=stop-start;k=self.in_features;weight=torch.empty((n,k),device=x.device,dtype=torch.bfloat16)
     _decode[(triton.cdiv(n,4),triton.cdiv(k,256))](self.codes[start:stop],self.scales[start:stop],self.zeros[start:stop],weight,n,k,4,256,enable_fp_fusion=False,num_warps=4)
     for i in range(count):output[i:i+1,start:stop]=torch.nn.functional.linear(inputs[i:i+1],weight)
     del weight
    return output.reshape(*shape,self.out_features)
   module.forward=types.MethodType(head,module)
 for layer in model.model.language_model.layers:
  router=layer.mlp.gate;original=router.forward
  def gate(self,x,_original=original):
   if x.ndim!=2 or not 2<=len(x)<=4:return _original(x)
   values=[_original(x[i:i+1]) for i in range(len(x))];assert all(isinstance(v,tuple) and len(v)==3 for v in values)
   return tuple(torch.cat([v[j] for v in values],0) for j in range(3))
  router.forward=types.MethodType(gate,router)
  shared_gate=layer.mlp.shared_expert_gate;original=shared_gate.forward
  def shared(self,x,_original=original):
   if x.ndim!=2 or not 2<=len(x)<=4:return _original(x)
   return torch.cat([_original(x[i:i+1]) for i in range(len(x))],0)
  shared_gate.forward=types.MethodType(shared,shared_gate)
