"""Preserve per-request norm and SDPA shapes as well as dense/router decode."""
import types
import torch
from split_decode122_native_math_v1 import bind as bind_previous

class PerRequestSDPARegistry:
 def __init__(self,original):self.original=original
 def get_interface(self,name,default):
  interface=self.original.get_interface(name,default)
  if name!='sdpa':return interface
  def attention(module,query,key,value,attention_mask,dropout=0.,scaling=None,is_causal=None,**kwargs):
   if query.shape[0]==1 or query.shape[2]!=1:return interface(module,query,key,value,attention_mask,dropout=dropout,scaling=scaling,is_causal=is_causal,**kwargs)
   assert 2<=query.shape[0]<=4 and key.shape[0]==value.shape[0]==query.shape[0]
   outputs=[]
   for i in range(query.shape[0]):
    mask=attention_mask if attention_mask is None or attention_mask.shape[0]==1 else attention_mask[i:i+1]
    output,weights=interface(module,query[i:i+1],key[i:i+1],value[i:i+1],mask,dropout=dropout,scaling=scaling,is_causal=is_causal,**kwargs)
    assert weights is None;outputs.append(output)
   return torch.cat(outputs,0),None
  return attention

def bind(model):
 bind_previous(model)
 norms=[model.model.language_model.norm]
 for layer in model.model.language_model.layers:
  norms.extend((layer.input_layernorm,layer.post_attention_layernorm))
  if hasattr(layer,'self_attn'):
   norms.extend((layer.self_attn.q_norm,layer.self_attn.k_norm))
   globals_=layer.self_attn.forward.__func__.__globals__
   assert type(globals_['ALL_ATTENTION_FUNCTIONS']).__name__=='_OwnAttentionRegistry'
   globals_['ALL_ATTENTION_FUNCTIONS']=PerRequestSDPARegistry(globals_['ALL_ATTENTION_FUNCTIONS'])
  else:
   norm=layer.linear_attn.norm;original=norm.forward;heads=layer.linear_attn.num_v_heads
   def gated(self,hidden_states,gate=None,_original=original,_heads=heads):
    if hidden_states.ndim!=2 or hidden_states.shape[0] not in (2*_heads,4*_heads):return _original(hidden_states,gate)
    assert gate is not None and gate.shape==hidden_states.shape and not torch.is_grad_enabled()
    return torch.cat([_original(hidden_states[i:i+_heads],gate[i:i+_heads]) for i in range(0,hidden_states.shape[0],_heads)],0)
   norm.forward=types.MethodType(gated,norm)
 for norm in norms:
  original=norm.forward
  def forward(self,x,_original=original):
   if x.ndim<3 or x.shape[1]!=1 or not 2<=x.shape[0]<=4:return _original(x)
   assert not torch.is_grad_enabled()
   return torch.cat([_original(x[i:i+1]) for i in range(x.shape[0])],0)
  norm.forward=types.MethodType(forward,norm)
