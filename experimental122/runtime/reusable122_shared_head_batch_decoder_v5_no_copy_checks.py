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
  for index,ids in enumerate(inputs):
   cache=StaticCache(self.model.config,max_cache_len=self.max_cache_len)
   assert not cache.offloading
   with torch.autocast('cuda',dtype=torch.bfloat16):
    output=self.model.model(input_ids=ids,past_key_values=cache,use_cache=True,
     position_ids=torch.arange(ids.shape[1],device='cuda')[None,None].expand(4,1,-1))
   last_states.append(output.last_hidden_state[:,-1:].clone())
   for target,source in zip(self.cache.layers,cache.layers):
    if type(target).__name__=='StaticLayer':
     length=ids.shape[1]
     target.keys[index:index+1,:,:length].copy_(source.keys[:,:,:length])
     target.values[index:index+1,:,:length].copy_(source.values[:,:,:length])
    else:
     assert type(target).__name__=='LinearAttentionLayer'
     target.conv_states[index:index+1].copy_(source.conv_states)
     target.recurrent_states[index:index+1].copy_(source.recurrent_states)
     target.has_previous_state=True
   del output,cache,source,target
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
