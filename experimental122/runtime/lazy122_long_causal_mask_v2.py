"""Materialize long single-request causal masks only for native SDPA row tiles."""
import ast,functools,hashlib,inspect,textwrap,types
import torch
from transformers import StaticCache
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as q

TEXT_FORWARD_SHA='e896ea60785115410f9ad92a787312efe291c6bec61418d6063d898600b84131'
STATS={'created':0,'slices':0,'maximum_slice_storage_bytes':0,'avoided_full_mask_bytes':0}

class LazyLongCausalMask:
 def __init__(self,positions,capacity):
  assert positions.ndim==2 and positions.shape[0]==1 and positions.shape[1]>4096
  assert torch.equal(positions,torch.arange(positions.shape[1],device=positions.device)[None])
  self.positions=positions;self.keys=torch.arange(capacity,device=positions.device)
  self.shape=(1,1,positions.shape[1],capacity)
  STATS['created']+=1;STATS['avoided_full_mask_bytes']=max(STATS['avoided_full_mask_bytes'],positions.shape[1]*capacity)
 def __getitem__(self,index):
  assert isinstance(index,tuple) and len(index)==4 and all(isinstance(v,slice) for v in index)
  assert index[0]==index[1]==index[3]==slice(None)
  rows=self.positions[:,index[2]];assert 1<=rows.shape[1]<=1024
  mask=rows[:,None,:,None]>=self.keys[None,None,None,:]
  assert mask.dtype==torch.bool and mask.is_contiguous()
  STATS['slices']+=1;STATS['maximum_slice_storage_bytes']=max(STATS['maximum_slice_storage_bytes'],mask.untyped_storage().nbytes())
  return mask

def lazy_create_causal_mask(config,inputs_embeds,attention_mask,past_key_values,position_ids=None,**kwargs):
 if inputs_embeds.shape[1]<=4096:
  return q.create_causal_mask(config,inputs_embeds,attention_mask,past_key_values,position_ids,**kwargs)
 assert not torch.is_grad_enabled() and inputs_embeds.shape[0]==1 and attention_mask is None and not kwargs
 assert config._attn_implementation=='sdpa' and getattr(config,'is_causal',True)
 assert isinstance(past_key_values,StaticCache) and int(past_key_values.get_seq_length())==0
 assert position_ids is not None
 full=next(layer for layer in past_key_values.layers if type(layer).__name__=='StaticLayer')
 return LazyLongCausalMask(position_ids,full.get_max_cache_shape())

def bind_lazy_long_mask_model(model):
 target=model.model.language_model;original=q.Qwen3_5MoeTextModel.forward
 assert type(target) is q.Qwen3_5MoeTextModel and target.forward.__func__ is original
 assert hashlib.sha256(inspect.getsource(original).strip().encode()).hexdigest()==TEXT_FORWARD_SHA
 tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
 assert sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='create_causal_mask' for n in ast.walk(tree))==1
 assert [ast.unparse(v) for v in tree.body[0].decorator_list]==['merge_with_config_defaults','capture_outputs','auto_docstring']
 raw=inspect.unwrap(original);namespace=dict(raw.__globals__);namespace['create_causal_mask']=lazy_create_causal_mask
 own=types.FunctionType(raw.__code__,namespace,raw.__name__,raw.__defaults__,raw.__closure__)
 functools.update_wrapper(own,raw);own.__kwdefaults__=raw.__kwdefaults__
 target.forward=types.MethodType(q.merge_with_config_defaults(q.capture_outputs(own)),target)
