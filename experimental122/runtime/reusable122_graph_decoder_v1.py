"""Reuse a single-request CUDA Graph while preserving fixed cache addresses."""
import torch
from gsq_graph_decoder_v3_cache_snapshot import GraphDecoder

class ReusableGraphDecoder(GraphDecoder):
 def prepare(self,ids):
  assert not torch.is_grad_enabled() and ids.ndim==2 and ids.shape[0]==1
  assert ids.shape[1]+4<self.max_cache_len
  if self.graph is None:
   super().prepare(ids)
   self._captured_cache_addresses=self.cache_addresses()
   self._captured_metadata=self.metadata()
   self.capture_count=getattr(self,'capture_count',0)+1
   self.reuse_count=0
   return
  assert self.cache_addresses()==self._captured_cache_addresses
  self.cache.reset()
  with torch.autocast('cuda',dtype=torch.bfloat16):
   output=self.model(input_ids=ids,past_key_values=self.cache,use_cache=True,position_ids=torch.arange(ids.shape[1],device='cuda')[None,None].expand(4,1,-1),logits_to_keep=1)
  self.token.copy_(output.logits[:,-1].argmax(-1,keepdim=True))
  self.position.fill_(ids.shape[1]);del output
  assert self.cache_addresses()==self._captured_cache_addresses
  assert self.metadata()==self._captured_metadata
  self.reuse_count+=1
 def cache_addresses(self):
  return tuple((i,n,v.data_ptr(),tuple(v.shape),str(v.dtype)) for i,layer in enumerate(self.cache.layers) for n,v in sorted(vars(layer).items()) if isinstance(v,torch.Tensor))
 def metadata(self):
  return tuple((i,n,v) for i,layer in enumerate(self.cache.layers) for n,v in sorted(vars(layer).items()) if isinstance(v,(bool,int,float,str)))
