"""Reuse fixed batch graph with independent body prefills and a shared head decode."""
import gc,hashlib,inspect
import torch
from transformers import StaticCache
from reusable122_batch_decoder_v3 import ReusableBatchDecoder as PreviousDecoder


class ReusableSharedHeadBatchDecoder(PreviousDecoder):
 def prepare(self,inputs):
  assert not torch.is_grad_enabled() and len(inputs)==self.token.shape[0]
  assert all(v.is_cuda and v.dtype==torch.long and v.ndim==2 and v.shape[0]==1
   and 0<v.shape[1]<self.max_cache_len-4 for v in inputs)
  assert self.addresses()==self._addresses
  assert hashlib.sha256(inspect.getsource(type(self.model).forward).encode()).hexdigest()=='e5ffa5b9ef79d82df426e9cdcfd38f532c7cd7a9983350683b0ed96ac561ef05'
  assert hashlib.sha256(inspect.getsource(type(self.model.model).forward).encode()).hexdigest()=='c23b3d81a6e5c036eb39e1f11cbddf6d77e9aa1b32b6aebf179bae7a58d29294'
  self.cache.reset();lengths=[v.shape[1] for v in inputs];last_states=[]
  # Prefill each body independently. Bound head reuses each decoded weight
  # block across requests, retaining the original per-row F.linear shape.
  # Keep only one temporary source cache and copy into the captured buffers.
  caches,last_states=layerwise_prefills(self.model,inputs,self.max_cache_len)
  for index,cache in enumerate(caches):
   for target,source in zip(self.cache.layers,cache.layers):
    if type(target).__name__=='StaticLayer':
     length=inputs[index].shape[1]
     target.keys[index:index+1,:,:length].copy_(source.keys[:,:,:length])
     target.values[index:index+1,:,:length].copy_(source.values[:,:,:length])
    else:
     assert type(target).__name__=='LinearAttentionLayer'
     target.conv_states[index:index+1].copy_(source.conv_states)
     target.recurrent_states[index:index+1].copy_(source.recurrent_states)
     target.has_previous_state=True
   del cache,source,target
  del caches
  for layer in self.cache.layers:
   if type(layer).__name__=='StaticLayer':layer.cumulative_length.fill_(max(lengths))
  with torch.autocast('cuda',dtype=torch.bfloat16):
   self.token.copy_(self.model.lm_head(torch.cat(last_states,0)).argmax(-1))
  del last_states
  self.shared_head_prepare_count=getattr(self,'shared_head_prepare_count',0)+1
  self.position.copy_(torch.tensor(lengths,device='cuda')[None,:,None].expand(4,-1,-1))
  assert self.addresses()==self._addresses and self.metadata()==self._metadata
  self.reuse_count+=1
  gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()


def layerwise_prefills(model,inputs,max_cache_len):
 from pathlib import Path
 from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as q
 from affine4_packed_native import PackedAffine4NativeLinear
 assert hashlib.sha256(Path(q.__file__).read_bytes()).hexdigest()=='f738a9eb46d1351a751f36a1cae5f7435d589e8c3218e7315182419c4c0230f1'
 assert not torch.is_grad_enabled() and len(inputs) in (2,4)
 language=model.model.language_model
 assert language.config.num_hidden_layers==48 and len(language.layers)==48
 caches=[];states=[];contexts=[]
 with torch.autocast('cuda',dtype=torch.bfloat16):
  for ids in inputs:
   cache=StaticCache(model.config,max_cache_len=max_cache_len);assert not cache.offloading
   hidden=language.embed_tokens(ids)
   position=torch.arange(ids.shape[1],device='cuda')[None,None].expand(4,1,-1)
   text_position=position[0];rope_position=position[1:]
   causal=q.create_causal_mask(config=language.config,inputs_embeds=hidden,attention_mask=None,past_key_values=cache,position_ids=text_position)
   linear=language._update_linear_attn_mask(None,cache)
   embeddings=language.rotary_emb(hidden,rope_position)
   caches.append(cache);states.append(hidden);contexts.append((embeddings,causal,linear,text_position))
  for index,layer in enumerate(language.layers):
   original=[];holders=[]
   try:
    for module in layer.modules():
     if isinstance(module,PackedAffine4NativeLinear):
      present='dequant' in vars(module);old=module.dequant;holder={}
      def cached(_old=old,_holder=holder):
       if 'weight' not in _holder:_holder['weight']=_old()
       return _holder['weight']
      original.append((module,old,present));holders.append(holder);module.dequant=cached
    for row,hidden in enumerate(states):
     embeddings,causal,linear,text_position=contexts[row]
     mask=linear if language.config.layer_types[index]=='linear_attention' else causal
     states[row]=layer(hidden,position_embeddings=embeddings,attention_mask=mask,
      position_ids=text_position,past_key_values=caches[row],use_cache=True)
   finally:
    for module,old,present in original:
     if present:module.dequant=old
     else:delattr(module,'dequant')
    for holder in holders:holder.clear()
   del original,holders
  last=[language.norm(hidden)[:,-1:].clone() for hidden in states]
 return caches,last
