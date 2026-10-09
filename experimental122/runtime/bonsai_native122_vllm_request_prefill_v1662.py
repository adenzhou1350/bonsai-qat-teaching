"""Experimental vLLM batch2-4 decode using validated native per-row arithmetic.

The projection/head/expert bodies retain their source AST. This adapter needs
actual whole-engine batch/cache checks before performance or deployment claims.
"""
import importlib.util,types
from pathlib import Path
import torch,triton
from affine4_packed_native import PackedAffine4NativeLinear
from primary_gsq122_dense4_runtime_v1 import PackedAffine4NativeChunkedHead
from affine4_packed_runtime import _decode
from native_lora_bmm_v1 import candidate as projection
from bonsai_native122_vllm_experts_v1564 import sha
from bonsai_native122_vllm_model_loader_v1593 import BonsaiNative122PackedTextForCausalLM,Native122PackedModelLoader
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.models.config import MODELS_CONFIG_MAP,Qwen3_5ForConditionalGenerationConfig
from vllm import ModelRegistry

ARCHITECTURE='BonsaiNative122RequestPrefillTextForCausalLM'
LOAD_FORMAT='bonsai_native122_request_prefill_packed_v1'
PINNED_PROVIDERS={
 'split_decode122_native_math_v1':'9d3a15a91e52101dcf2b34a398dc9eeb3eee276834b357b41fd207fafd3ec9a2',
 'small_batch122_experts_v2':'7020398cc0538e8b23480cccd6c26d0047caaaa85b79bcf5677be96aa9500f3c',
 'native_lora_bmm_v1':'230adc4c0ffbfd815e1afbf93117450c4ec54028acab02d5127346e08f2ed5f0',
 'bonsai_native122_vllm_model_loader_v1593':'ac6a7ea0ed1eddcaea95be91d8c592c20c2dc95c9a6a0d5fdb8c3db6c62aa40e',
}
assert all(sha(Path(importlib.util.find_spec(n).origin))==h for n,h in PINNED_PROVIDERS.items())

def bind_vllm_small_decode(model):
 counts={'dense_parts':0,'heads':0,'expert_banks':0,'routers_and_shared_gates':0}
 for module in model.modules():
  if isinstance(module,PackedAffine4NativeLinear):
   original=module.forward
   def linear(self, x, _original=original):
       count = x.numel() // self.in_features
       if not 2 <= count <= 4:
           return _original(x)
       assert not torch.is_grad_enabled()
       shape = x.shape[:-1]
       inputs = x.to(torch.bfloat16).reshape(-1, self.in_features).contiguous()
       weight = self.dequant()
       return torch.cat([torch.nn.functional.linear(inputs[i:i + 1], weight, self.bias if self.has_bias else None) for i in range(count)], 0).reshape(*shape, self.out_features)
   module.forward=types.MethodType(linear,module);counts['dense_parts']+=1
  elif isinstance(module,PackedAffine4NativeChunkedHead):
   original=module.forward
   def head(self, x, _original=original):
       count = x.numel() // self.in_features
       if not 2 <= count <= 4:
           return _original(x)
       assert not torch.is_grad_enabled() and (not self.has_bias)
       shape = x.shape[:-1]
       inputs = x.to(torch.bfloat16).reshape(-1, self.in_features).contiguous()
       output = torch.empty((count, self.out_features), device=x.device, dtype=torch.bfloat16)
       for start in range(0, self.out_features, 2048):
           stop = min(start + 2048, self.out_features)
           n = stop - start
           k = self.in_features
           weight = torch.empty((n, k), device=x.device, dtype=torch.bfloat16)
           _decode[triton.cdiv(n, 4), triton.cdiv(k, 256)](self.codes[start:stop], self.scales[start:stop], self.zeros[start:stop], weight, n, k, 4, 256, enable_fp_fusion=False, num_warps=4)
           for i in range(count):
               output[i:i + 1, start:stop] = torch.nn.functional.linear(inputs[i:i + 1], weight)
           del weight
       return output.reshape(*shape, self.out_features)
   module.forward=types.MethodType(head,module);counts['heads']+=1
 for routed in model.modules():
  if not isinstance(routed,RoutedExperts):continue
  module=routed.bonsai_experts;original=module.forward
  def forward(self, hidden, selected, probabilities, _original=original):
      if not 2 <= len(hidden) <= 8:
          return _original(hidden, selected, probabilities)
      assert not torch.is_grad_enabled() and selected.shape == probabilities.shape and (selected.shape[-1] == 8)
      x = hidden.to(torch.bfloat16).contiguous()
      fanout = 8
      ids, order = selected.reshape(-1).sort()
      routed = x[order // fanout].contiguous()
      gate, up = projection(self.gate, routed, ids).chunk(2, -1)
      middle = (torch.nn.functional.silu(gate) * up).contiguous()
      result = projection(self.down, middle, ids)
      weighted = result * probabilities.reshape(-1)[order, None]
      inverse = torch.empty_like(order)
      inverse[order] = torch.arange(len(order), device=order.device)
      return weighted[inverse].reshape(len(hidden), fanout, -1).sum(1).to(hidden.dtype)
  module.forward=types.MethodType(forward,module);counts['expert_banks']+=1
 for layer in model.model.layers:
  for module in (layer.mlp.gate,layer.mlp.shared_expert_gate):
   original=module.forward
   def scalar_rows(self,x,_original=original):
    if x.ndim!=2 or not 2<=len(x)<=4:return _original(x)
    pieces=[_original(x[i:i+1]) for i in range(len(x))]
    assert all(isinstance(v,tuple) and len(v)==2 and v[1] is None for v in pieces)
    return torch.cat([v[0] for v in pieces],0),None
   module.forward=types.MethodType(scalar_rows,module);counts['routers_and_shared_gates']+=1
 assert counts=={'dense_parts':372,'heads':1,'expert_banks':48,'routers_and_shared_gates':96},counts
 return {'bound_modules':counts,'pinned_provider_SHA256':PINNED_PROVIDERS,'dense_head_expert_bodies_AST_unchanged':True,'lowrank_BMM_only_projection':True,'stock_vllm_attention_GDN_norms_retained':True}


# Preserve each prompt's original prefill GEMM shape, including multiple new
# prompts admitted in the same mixed batch. Copy sequence boundaries once per
# eager prefill context; pure decode graphs do not read CUDA metadata on CPU.
from vllm.forward_context import get_forward_context
from embedding4_packed_runtime import PackedAffine4Embedding
MIXED_AUDIT={'dense_head_calls':0,'expert_calls':0,'router_calls':0,'embedding_calls':0,'prefill_context_CPU_boundary_reads':0,'layouts':{}}
_PREFILL_CONTEXT=None
_PREFILL_SLICES=None

def request_prefill_slices(x):
 global _PREFILL_CONTEXT,_PREFILL_SLICES
 if len(x)<=4:return None
 context=get_forward_context();metadata=context.attn_metadata
 if metadata is None:return None
 assert isinstance(metadata,dict)
 values=[v for v in metadata.values() if hasattr(v,'num_decode_tokens') and hasattr(v,'num_prefill_tokens')]
 if not values or not values[0].num_prefill_tokens:return None
 if context is not _PREFILL_CONTEXT:
  first=values[0];assert first.spec_sequence_masks is None
  assert first.num_decode_tokens+first.num_prefill_tokens==len(x)
  assert all(v.num_decode_tokens==first.num_decode_tokens and v.num_prefill_tokens==first.num_prefill_tokens for v in values)
  boundaries=first.non_spec_query_start_loc.detach().cpu().tolist()
  assert boundaries[0]==0 and boundaries[-1]==len(x) and all(b>a for a,b in zip(boundaries,boundaries[1:]))
  assert len(boundaries)==first.num_decodes+first.num_prefills+1
  _PREFILL_CONTEXT=context;_PREFILL_SLICES=list(zip(boundaries,boundaries[1:]))
  MIXED_AUDIT['prefill_context_CPU_boundary_reads']+=1
  key='+'.join(str(b-a) for a,b in _PREFILL_SLICES);MIXED_AUDIT['layouts'][key]=MIXED_AUDIT['layouts'].get(key,0)+1
 assert _PREFILL_SLICES[-1][1]==len(x)
 return _PREFILL_SLICES

def bind_request_prefill_boundaries(model):
 counts={'dense_head':0,'experts':0,'routers':0,'embedding':0}
 for module in model.modules():
  if isinstance(module,(PackedAffine4NativeLinear,PackedAffine4NativeChunkedHead)):
   original=module.forward
   def dense(self,x,_original=original):
    flat=x.reshape(-1,self.in_features);slices=request_prefill_slices(flat)
    if not slices:return _original(x)
    MIXED_AUDIT['dense_head_calls']+=1
    return torch.cat([_original(flat[a:b]) for a,b in slices],0).reshape(*x.shape[:-1],self.out_features)
   module.forward=types.MethodType(dense,module);counts['dense_head']+=1
  elif isinstance(module,PackedAffine4Embedding):
   original=module.forward
   def embedding(self,ids,_original=original):
    flat=ids.reshape(-1);slices=request_prefill_slices(flat)
    if not slices:return _original(ids)
    MIXED_AUDIT['embedding_calls']+=1
    return torch.cat([_original(flat[a:b]) for a,b in slices],0).reshape(*ids.shape,self.embedding_dim)
   module.forward=types.MethodType(embedding,module);counts['embedding']+=1
  elif isinstance(module,RoutedExperts):
   experts=module.bonsai_experts;original=experts.forward
   def expert(self,hidden,selected,probabilities,_original=original):
    slices=request_prefill_slices(hidden)
    if not slices:return _original(hidden,selected,probabilities)
    MIXED_AUDIT['expert_calls']+=1
    return torch.cat([_original(hidden[a:b],selected[a:b],probabilities[a:b]) for a,b in slices],0)
   experts.forward=types.MethodType(expert,experts);counts['experts']+=1
 for layer in model.model.layers:
  for module in (layer.mlp.gate,layer.mlp.shared_expert_gate):
   original=module.forward
   def router(self,x,_original=original):
    slices=request_prefill_slices(x)
    if not slices:return _original(x)
    MIXED_AUDIT['router_calls']+=1
    values=[_original(x[a:b]) for a,b in slices];assert all(v[1] is None for v in values)
    return torch.cat([v[0] for v in values],0),None
   module.forward=types.MethodType(router,module);counts['routers']+=1
 assert counts=={'dense_head':373,'experts':48,'routers':96,'embedding':1},counts
 return counts

class BonsaiNative122BatchedTextForCausalLM(BonsaiNative122PackedTextForCausalLM):
 def __init__(self,*,vllm_config,prefix=''):
  super().__init__(vllm_config=vllm_config,prefix=prefix)
  self.bonsai_small_decode_binding=bind_vllm_small_decode(self)
  self.bonsai_small_decode_binding['mixed_decode_prefix_binding']=bind_request_prefill_boundaries(self)

@register_model_loader(LOAD_FORMAT)
class Native122BatchedPackedModelLoader(Native122PackedModelLoader):
 def load_weights(self,model,model_config):
  super().load_weights(model,model_config)
  import json
  p=Path(self.load_config.model_loader_extra_config['weight_proof']);v=json.loads(p.read_text());v['small_decode_binding']=model.bonsai_small_decode_binding;p.write_text(json.dumps(v,indent=2))

def register_model():
 MODELS_CONFIG_MAP[ARCHITECTURE]=Qwen3_5ForConditionalGenerationConfig
 ModelRegistry.register_model(ARCHITECTURE,BonsaiNative122BatchedTextForCausalLM)
