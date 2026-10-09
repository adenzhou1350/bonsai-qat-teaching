"""Reuse one fixed B2/B4 graph, replacing prefixes at captured addresses."""
import gc
import torch
from transformers import StaticCache
from variable_prefill122_batch_decoder_v2_shared_stream import VariablePrefillBatchDecoder


class ReusableBatchDecoder(VariablePrefillBatchDecoder):
 def __init__(self,model,inputs,max_cache_len=1536):
  super().__init__(model,inputs,max_cache_len)
  self._addresses=self.addresses();self._metadata=self.metadata()
  self.capture_count=1;self.reuse_count=0

 def addresses(self):
  return (self.token.data_ptr(),self.position.data_ptr(),self.keys.data_ptr(),tuple(
   (i,n,v.data_ptr(),tuple(v.shape),str(v.dtype))
   for i,layer in enumerate(self.cache.layers)
   for n,v in sorted(vars(layer).items()) if isinstance(v,torch.Tensor)))

 def metadata(self):
  return tuple((i,n,v) for i,layer in enumerate(self.cache.layers)
   for n,v in sorted(vars(layer).items()) if isinstance(v,(bool,int,float,str)))

 def prepare(self,inputs):
  assert not torch.is_grad_enabled() and len(inputs)==self.token.shape[0]
  assert all(v.is_cuda and v.dtype==torch.long and v.ndim==2 and v.shape[0]==1
   and 0<v.shape[1]<self.max_cache_len-4 for v in inputs)
  assert self.addresses()==self._addresses
  self.cache.reset();lengths=[v.shape[1] for v in inputs]
  # Prefill each request independently, with the original B1 operator shapes.
  # Keep only one temporary source cache and copy into the captured buffers.
  for index,ids in enumerate(inputs):
   cache=StaticCache(self.model.config,max_cache_len=self.max_cache_len)
   assert not cache.offloading
   with torch.autocast('cuda',dtype=torch.bfloat16):
    output=self.model(input_ids=ids,past_key_values=cache,use_cache=True,
     position_ids=torch.arange(ids.shape[1],device='cuda')[None,None].expand(4,1,-1),logits_to_keep=1)
   self.token[index:index+1].copy_(output.logits[:,-1].argmax(-1,keepdim=True))
   for target,source in zip(self.cache.layers,cache.layers):
    if type(target).__name__=='StaticLayer':
     length=ids.shape[1]
     target.keys[index:index+1,:,:length].copy_(source.keys[:,:,:length])
     target.values[index:index+1,:,:length].copy_(source.values[:,:,:length])
     assert torch.equal(target.keys[index:index+1,:,:length],source.keys[:,:,:length])
     assert torch.equal(target.values[index:index+1,:,:length],source.values[:,:,:length])
    else:
     assert type(target).__name__=='LinearAttentionLayer'
     target.conv_states[index:index+1].copy_(source.conv_states)
     target.recurrent_states[index:index+1].copy_(source.recurrent_states)
     target.has_previous_state=True
     assert torch.equal(target.conv_states[index:index+1],source.conv_states)
     assert torch.equal(target.recurrent_states[index:index+1],source.recurrent_states)
   del output,cache,source,target
  for layer in self.cache.layers:
   if type(layer).__name__=='StaticLayer':layer.cumulative_length.fill_(max(lengths))
  self.position.copy_(torch.tensor(lengths,device='cuda')[None,:,None].expand(4,-1,-1))
  assert self.addresses()==self._addresses and self.metadata()==self._metadata
  self.reuse_count+=1
  gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
