"""Graph-compatible selected-route decode for up to eight token rows."""
import hashlib,inspect,types
import torch

def projection(bank,inputs,ids):
 assert not torch.is_grad_enabled() and inputs.dtype==torch.bfloat16
 routes=len(ids);e,n,k=bank.shape
 assert ids.ndim==1 and inputs.shape==(routes,k) and 1<=routes<=64
 weights=bank.weights(ids);base=torch.empty((routes,n),device=inputs.device,dtype=inputs.dtype)
 for route in range(routes):torch.mm(inputs[route:route+1],weights[route].transpose(-2,-1),out=base[route:route+1])
 del weights
 a=bank.lora_a[ids];b=bank.lora_b[ids]
 low=torch.empty((routes,8),device=inputs.device,dtype=inputs.dtype)
 for route in range(routes):torch.mm(inputs[route:route+1],a[route].transpose(-2,-1),out=low[route:route+1])
 del a
 correction=torch.empty_like(base)
 for route in range(routes):torch.mm(low[route:route+1],b[route].transpose(-2,-1),out=correction[route:route+1])
 return (base+correction*bank.lora_scale).to(torch.bfloat16)

def bind(model):
 from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as q
 assert hashlib.sha256(inspect.getsource(q.Qwen3_5MoeRMSNorm.forward).strip().encode()).hexdigest()=='028dcc012a97674063c6d9e57f99246d081a9af06307da252fe551caacab1899'
 for layer in model.model.language_model.layers:
  if hasattr(layer,'self_attn'):
   for norm in (layer.self_attn.q_norm,layer.self_attn.k_norm):
    original_norm=norm.forward
    def norm_forward(self,x,_original=original_norm):
     if x.shape[0]==1:return _original(x)
     assert not torch.is_grad_enabled() and x.ndim==4 and 2<=x.shape[0]<=4 and x.shape[1]<=4096 and x.shape[-1]==256 and x.dtype==torch.bfloat16
     return q.Qwen3_5MoeRMSNorm.forward(self,x)
    norm.forward=types.MethodType(norm_forward,norm)
 for layer in model.model.language_model.layers:
  module=layer.mlp.experts;original=module.forward
  def forward(self,hidden,selected,probabilities,_original=original):
   if not 2<=len(hidden)<=8:return _original(hidden,selected,probabilities)
   assert not torch.is_grad_enabled() and selected.shape==probabilities.shape and selected.shape[-1]==8
   x=hidden.to(torch.bfloat16).contiguous();fanout=8;ids,order=selected.reshape(-1).sort()
   routed=x[order//fanout].contiguous()
   gate,up=projection(self.gate,routed,ids).chunk(2,-1)
   middle=(torch.nn.functional.silu(gate)*up).contiguous()
   result=projection(self.down,middle,ids)
   weighted=result*probabilities.reshape(-1)[order,None]
   inverse=torch.empty_like(order);inverse[order]=torch.arange(len(order),device=order.device)
   return weighted[inverse].reshape(len(hidden),fanout,-1).sum(1).to(hidden.dtype)
  module.forward=types.MethodType(forward,module)
