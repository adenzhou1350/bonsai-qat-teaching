"""Inference dependency extracted from validated experimental runtime."""

import ast, hashlib, inspect, textwrap, types

import torch

from transformers.integrations import sdpa_attention as s

from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as q

ATTENTION_FORWARD_SHA='d01fa152f83e1c535a5d6387f31d440d90977fda0c0380894ebb74721dbec2ca'

SDPA_FORWARD_SHA='a5e1a66fdaa0e143a539415fc593cecd7f5e48213c9aebf0c8f0c4635cdd25bb'

def bounded_head_sdpa(module,query,key,value,attention_mask,dropout=0.0,scaling=None,is_causal=None,**kwargs):
 if query.shape[2]<=4096 or attention_mask is None:
  return s.sdpa_attention_forward(module,query,key,value,attention_mask,dropout=dropout,scaling=scaling,is_causal=is_causal,**kwargs)
 assert not torch.is_grad_enabled() and not module.training and dropout==0.0 and not kwargs.get('output_attentions',False)
 assert query.shape[0]==key.shape[0]==value.shape[0]==1 and query.shape[1]==32 and key.shape[1]==value.shape[1]==2 and query.shape[-1]==256 and query.dtype==key.dtype==value.dtype==torch.bfloat16
 assert module.num_key_value_groups==16
 output=torch.empty((query.shape[0],query.shape[2],query.shape[1],query.shape[3]),device=query.device,dtype=query.dtype)
 proxy=types.SimpleNamespace(num_key_value_groups=1,is_causal=module.is_causal)
 for head in range(0,32,1):
  kv=head//16
  # One head needs one KV view. contiguous() preserves shape, values and native SDPA;
  # avoid repeat_interleave(1), which allocates two redundant full-prefix copies.
  keys=key[:,kv:kv+1].contiguous()
  values=value[:,kv:kv+1].contiguous()
  assert attention_mask.shape[1]==1 and is_causal in (None,False)
  # Preserve all keys and the explicit absolute-position mask for every query.
  # Native SDPA computes bounded query rows; no context, head, precision or
  # attention probability is dropped. Whole numerical admission still required.
  for start in range(0,query.shape[2],1024):
   end=min(start+1024,query.shape[2])
   chunk,weights=s.sdpa_attention_forward(proxy,query[:,head:head+1,start:end],keys,values,attention_mask[:,:,start:end,:],dropout=dropout,scaling=scaling,is_causal=False,**kwargs)
   assert weights is None and chunk.dtype==output.dtype
   output[:,start:end,head:head+1].copy_(chunk)
   del chunk
  del keys,values
 return output,None

class _OwnAttentionRegistry:
 def __init__(self,original):self.original=original
 def get_interface(self,name,default):
  interface=self.original.get_interface(name,default)
  if name=='sdpa':
   assert interface is s.sdpa_attention_forward
   return bounded_head_sdpa
  return interface

def bounded_attention_output_gate(attn_output,gate):
 if attn_output.shape[1]<=4096:return attn_output*torch.sigmoid(gate)
 assert not torch.is_grad_enabled() and attn_output.shape==gate.shape and attn_output.ndim==3 and attn_output.shape[0]==1 and attn_output.shape[-1]==8192 and attn_output.dtype==gate.dtype==torch.bfloat16
 assert attn_output.is_contiguous() and attn_output.untyped_storage().data_ptr()!=gate.untyped_storage().data_ptr()
 # The owned long SDPA output is fresh, exclusive storage. Keep native BF16
 # sigmoid rounding, followed by native BF16 multiply rounding, in bounded rows.
 for start in range(0,attn_output.shape[1],1024):
  end=min(start+1024,attn_output.shape[1]);sigmoid=torch.sigmoid(gate[:,start:end])
  attn_output[:,start:end].mul_(sigmoid);del sigmoid
 return attn_output

def own_attention_forward():
 original=q.Qwen3_5MoeAttention.forward
 assert hashlib.sha256(inspect.getsource(original).strip().encode()).hexdigest()==ATTENTION_FORWARD_SHA
 assert hashlib.sha256(inspect.getsource(s.sdpa_attention_forward).strip().encode()).hexdigest()==SDPA_FORWARD_SHA
 tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
 target=ast.parse('attn_output = attn_output * torch.sigmoid(gate)').body[0]
 replacement=ast.parse('attn_output = bounded_attention_output_gate(attn_output, gate)').body[0]
 found=0
 for index,node in enumerate(tree.body[0].body):
  if ast.dump(node,include_attributes=False)==ast.dump(target,include_attributes=False):
   tree.body[0].body[index]=replacement;found+=1
 assert found==1 and original.__closure__ is None
 namespace=dict(original.__globals__);namespace['bounded_attention_output_gate']=bounded_attention_output_gate
 namespace['ALL_ATTENTION_FUNCTIONS']=_OwnAttentionRegistry(q.ALL_ATTENTION_FUNCTIONS)
 exec(compile(ast.fix_missing_locations(tree),'<owned-native-attention-bounded-gate>','exec'),namespace)
 own=namespace['forward'];own.__defaults__=original.__defaults__;own.__kwdefaults__=original.__kwdefaults__
 return own

def bind_bounded_head_sdpa_layer(layer):
 if hasattr(layer,'self_attn'):
  attn=layer.self_attn;assert type(attn) is q.Qwen3_5MoeAttention
  attn.forward=types.MethodType(own_attention_forward(),attn)

def bind_bounded_head_sdpa_model(model):
 for layer in model.model.language_model.layers:bind_bounded_head_sdpa_layer(layer)
