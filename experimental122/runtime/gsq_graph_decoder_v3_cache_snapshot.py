"""Inference dependency extracted from validated experimental runtime."""

import gc, hashlib, inspect

import torch

from transformers import StaticCache

class GraphDecoder:
 def __init__(self,model,prototype,max_cache_len=8192):
  self.model=model;self.max_cache_len=max_cache_len
  self.cache=StaticCache(model.config,max_cache_len=max_cache_len)
  assert self.cache.offloading is False
  self.token=torch.zeros((1,1),device='cuda',dtype=torch.long)
  self.position=torch.zeros((4,1,1),device='cuda',dtype=torch.long)
  self.graph=None;self.lifecycle_events=[]
  self.prepare(prototype)
 def step(self):
  with torch.autocast('cuda',dtype=torch.bfloat16):
   output=self.model(input_ids=self.token,past_key_values=self.cache,use_cache=True,position_ids=self.position,logits_to_keep=1)
  self.token.copy_(output.logits[:,-1].argmax(-1,keepdim=True));self.position.add_(1)
 def prepare(self,ids):
  assert not torch.is_grad_enabled() and ids.ndim==2 and ids.shape[0]==1
  assert ids.shape[1]+4<self.max_cache_len
  torch.cuda.synchronize();before=torch.cuda.memory_allocated()
  if self.graph is not None:
   old=self.graph;self.graph=None;old.reset();del old
   gc.collect();torch.cuda.empty_cache()
  after=torch.cuda.memory_allocated()
  self.cache.reset()
  with torch.autocast('cuda',dtype=torch.bfloat16):
   output=self.model(input_ids=ids,past_key_values=self.cache,use_cache=True,position_ids=torch.arange(ids.shape[1],device='cuda')[None,None].expand(4,1,-1),logits_to_keep=1)
  self.token.copy_(output.logits[:,-1].argmax(-1,keepdim=True));self.position.fill_(ids.shape[1]);del output
  snapshots=[];metadata=[];classes={}
  for layer in self.cache.layers:
   cls=type(layer);classes[cls.__name__]=hashlib.sha256(inspect.getsource(cls).encode()).hexdigest()
   required={'StaticLayer':{'keys','values','cumulative_length'},'LinearAttentionLayer':{'conv_states','recurrent_states'}}[cls.__name__]
   actual={n for n,v in vars(layer).items() if isinstance(v,torch.Tensor)}
   assert actual==required,(cls.__name__,actual)
   for name in sorted(actual):
    value=getattr(layer,name);assert value.is_cuda
    snapshots.append((layer,name,value,value.clone()))
   metadata.append((layer,{n:v for n,v in vars(layer).items() if isinstance(v,(bool,int,float,str))}))
  token=self.token.clone();position=self.position.clone()
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for i in range(3):self.step()
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
  self.graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(self.graph):self.step()
  # Copy into the exact captured addresses, including cumulative length and
  # full linear recurrent state. Never replace buffers or move cache to CPU.
  for layer,name,value,saved in snapshots:
   assert getattr(layer,name) is value
   value.copy_(saved);assert torch.equal(value,saved)
  for layer,values in metadata:
   assert all(getattr(layer,n)==v for n,v in values.items())
  self.token.copy_(token);self.position.copy_(position)
  self.lifecycle_events.append({'context_tokens':int(ids.shape[1]),'allocated_before_graph_release':before,
   'allocated_after_graph_release':after,'GPU_cache_snapshot_bytes':sum(v.numel()*v.element_size() for _,_,_,v in snapshots),
   'all_cache_tensor_addresses_and_values_restored':True,'token_and_position_restored':True,
   'cache_class_sha256':classes,'cache_offloading':False})
  del snapshots,metadata,token,position;gc.collect();torch.cuda.synchronize()
 def generate_ids(self,ids,cap,eos):
  assert ids.shape[1]+cap<=self.max_cache_len
  self.prepare(ids);result=[]
  for i in range(cap):
   if i:self.graph.replay()
   value=int(self.token.item());result.append(value)
   if value in eos:break
  return result
