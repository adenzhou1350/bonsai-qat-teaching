"""Fixed2/4-request microbatch with independent lengths, cache writes and EOS caller."""
import gc,hashlib,inspect,types
import torch
from transformers import StaticCache
from transformers.cache_utils import StaticLayer

_SHARED_STREAMS={}
def get_shared_stream():
 device=torch.cuda.current_device()
 if device not in _SHARED_STREAMS:_SHARED_STREAMS[device]=torch.cuda.Stream(device=device)
 return _SHARED_STREAMS[device]

UPDATE_SHA='14d004ef66e52358531606fe7fa7a95fbf1f57e470f2ab9af7c46d4eeb1a264c'

def bind_variable_static_update(target,positions):
 assert type(target) is StaticLayer and target.is_initialized
 assert hashlib.sha256(inspect.getsource(StaticLayer.update).encode()).hexdigest()==UPDATE_SHA
 assert positions.ndim==1 and positions.shape[0] in (2,4) and positions.dtype==torch.long
 def update(self,key_states,value_states,*args,**kwargs):
  assert self.is_initialized and not args and not kwargs
  assert key_states.shape[0]==value_states.shape[0]==positions.shape[0] and key_states.shape[-2]==value_states.shape[-2]==1
  for index in range(positions.shape[0]):
   self.keys[index:index+1].index_copy_(2,positions[index:index+1],key_states[index:index+1])
   self.values[index:index+1].index_copy_(2,positions[index:index+1],value_states[index:index+1])
  self.cumulative_length.add_(1)
  return self.keys,self.values
 target.update=types.MethodType(update,target)

class VariablePrefillBatchDecoder:
 def __init__(self,model,inputs,max_cache_len=1536):
  assert not torch.is_grad_enabled() and isinstance(inputs,list) and len(inputs) in (2,4)
  assert all(v.is_cuda and v.dtype==torch.long and v.ndim==2 and v.shape[0]==1 and 0<v.shape[1]<max_cache_len-4 for v in inputs)
  self.model=model;self.max_cache_len=max_cache_len;self.cache=StaticCache(model.config,max_cache_len=max_cache_len)
  assert not self.cache.offloading
  batch=len(inputs);self.token=torch.zeros((batch,1),device='cuda',dtype=torch.long)
  self.position=torch.zeros((4,batch,1),device='cuda',dtype=torch.long);self.keys=torch.arange(max_cache_len,device='cuda')
  self.graph=None;self.lifecycle_events=[];sources=[];tokens=[]
  lengths=[v.shape[1] for v in inputs]
  for ids in inputs:
   cache=StaticCache(model.config,max_cache_len=max_cache_len);assert not cache.offloading
   with torch.autocast('cuda',dtype=torch.bfloat16):output=model(input_ids=ids,past_key_values=cache,use_cache=True,position_ids=torch.arange(ids.shape[1],device='cuda')[None,None].expand(4,1,-1),logits_to_keep=1)
   sources.append(cache);tokens.append(output.logits[:,-1].argmax(-1,keepdim=True));del output
  for li,target in enumerate(self.cache.layers):
   parents=[cache.layers[li] for cache in sources]
   if type(target).__name__=='StaticLayer':
    source=parents[0]
    target.lazy_initialization(torch.empty((batch,source.keys.shape[1],1,source.keys.shape[-1]),device='cuda',dtype=source.keys.dtype),torch.empty((batch,source.values.shape[1],1,source.values.shape[-1]),device='cuda',dtype=source.values.dtype))
    for i,(source,length) in enumerate(zip(parents,lengths)):
     target.keys[i:i+1,:,:length].copy_(source.keys[:,:,:length]);target.values[i:i+1,:,:length].copy_(source.values[:,:,:length])
     assert torch.equal(target.keys[i:i+1,:,:length],source.keys[:,:,:length]) and torch.equal(target.values[i:i+1,:,:length],source.values[:,:,:length])
    target.cumulative_length.fill_(max(lengths));bind_variable_static_update(target,self.position[0,:,0])
   else:
    assert type(target).__name__=='LinearAttentionLayer'
    conv=torch.cat([source.conv_states for source in parents],0);state=torch.cat([source.recurrent_states for source in parents],0)
    self.cache.update_conv_state(conv,li);self.cache.update_recurrent_state(state,li)
    assert torch.equal(target.conv_states,conv) and torch.equal(target.recurrent_states,state) and target.has_previous_state
    del conv,state
  self.token.copy_(torch.cat(tokens,0));self.position.copy_(torch.tensor(lengths,device='cuda')[None,:,None].expand(4,-1,-1))
  del sources,parents,tokens,source,cache;gc.collect();torch.cuda.empty_cache()
  snapshots=[];metadata=[]
  for layer in self.cache.layers:
   required={'StaticLayer':{'keys','values','cumulative_length'},'LinearAttentionLayer':{'conv_states','recurrent_states'}}[type(layer).__name__]
   actual={n for n,v in vars(layer).items() if isinstance(v,torch.Tensor)};assert actual==required
   for name in sorted(actual):
    value=getattr(layer,name);assert value.is_cuda;snapshots.append((layer,name,value,value.clone()))
   metadata.append((layer,{n:v for n,v in vars(layer).items() if isinstance(v,(bool,int,float,str))}))
  saved_token=self.token.clone();saved_position=self.position.clone()
  stream=get_shared_stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(3):self.step()
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
  self.graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(self.graph,stream=stream):self.step()
  torch.cuda.current_stream().wait_stream(stream)
  for layer,name,value,saved in snapshots:
   assert getattr(layer,name) is value;value.copy_(saved);assert torch.equal(value,saved)
  assert all(all(getattr(layer,n)==v for n,v in values.items()) for layer,values in metadata)
  self.token.copy_(saved_token);self.position.copy_(saved_position)
  self.lifecycle_events.append({'input_lengths':lengths,'per_request_KV_writes':True,'per_request_absolute_RoPE_positions':True,'per_request_boolean_causal_masks':True,'GPU_cache_snapshot_bytes':sum(v.numel()*v.element_size() for _,_,_,v in snapshots),'all_cache_addresses_and_values_restored':True,'cache_offloading':False})
  del snapshots,metadata,saved_token,saved_position;gc.collect();torch.cuda.synchronize()
 def step(self):
  # An explicit4D mask is passed through Transformers unchanged. There is no padding input propagation.
  mask=self.position[0,:,0,None,None,None]>=self.keys[None,None,None,:]
  assert mask.dtype==torch.bool and mask.shape==(self.token.shape[0],1,1,self.max_cache_len)
  with torch.autocast('cuda',dtype=torch.bfloat16):output=self.model(input_ids=self.token,past_key_values=self.cache,use_cache=True,position_ids=self.position,attention_mask=mask,logits_to_keep=1)
  self.token.copy_(output.logits[:,-1].argmax(-1,keepdim=True));self.position.add_(1)
