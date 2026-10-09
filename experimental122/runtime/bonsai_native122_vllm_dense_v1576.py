"""Native122 dense4/embedding adapter; preserve source projection boundaries.

Expert adaptation is supplied by bonsai_native122_vllm_experts_v1564.
This module alone is not a full vLLM model loader or engine validation.
"""
import json,re
from pathlib import Path
from bonsai_native122_vllm_experts_v1564 import Native122ExpertsConfig,Native122ExpertMethod,sha
import torch
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.linear import LinearBase,LinearMethodBase,UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding,ParallelLMHead
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from affine4_packed_native import PackedAffine4NativeLinear
from primary_gsq122_dense4_runtime_v1 import PackedAffine4NativeChunkedHead
from embedding4_packed_runtime import PackedAffine4Embedding

NAME='bonsai_native122_compact_control_v1'
PINNED_ADDITIONAL_SOURCE={
 'model_executor/layers/linear.py':'0bd60103d85d7183791077c46e0770d0a2adc8e8811b162b9084c3cbe92ba662',
 'model_executor/layers/vocab_parallel_embedding.py':'90bc72a4758c44fd631dda30a5e8695369bf6e3fe5e1d9b7edc80356abfcd110',
 'model_executor/models/qwen3_5.py':'86ca256c5e3803245f27ac5c8d931cd3d3cb21611aaf2ba90961fb72646c5ca6',
 'model_executor/models/qwen3_next.py':'885ce3244d069871bd8df93e67d94e4f394259c951f6a778947743d4a57b7fa5',
}
def check_additional_contract():
 import importlib.util
 base=Path(importlib.util.find_spec('vllm').origin).parent
 assert all(sha(base/n)==v for n,v in PINNED_ADDITIONAL_SOURCE.items())
check_additional_contract()
def canonical_prefix(prefix):
 if prefix.startswith('model.language_model.'):return prefix[len('model.language_model.'):]
 if prefix.startswith('model.'):return prefix[len('model.'):]
 return prefix
def dense_plan(manifest):
 rows={v['module']:v for v in manifest['matrices'] if v['packing_format']=='own-affine-dense4-v1'};assert len(rows)==373
 plans={};consumed=set()
 def add(target,sources):
  assert target not in plans and all(n in rows and n not in consumed for n in sources)
  values=[rows[n] for n in sources];assert len({v['shape'][1] for v in values})==1
  plans[target]=values;consumed.update(sources)
 for li in range(48):
  p=f'layers.{li}'
  if li%4==3:add(p+'.self_attn.qkv_proj',[p+'.self_attn.'+n+'_proj' for n in ('q','k','v')])
  else:
   add(p+'.linear_attn.in_proj_qkvz',[p+'.linear_attn.in_proj_qkv',p+'.linear_attn.in_proj_z'])
   add(p+'.linear_attn.in_proj_ba',[p+'.linear_attn.in_proj_b',p+'.linear_attn.in_proj_a'])
  add(p+'.mlp.shared_expert.gate_up_proj',[p+'.mlp.shared_expert.gate_proj',p+'.mlp.shared_expert.up_proj'])
 for name in sorted(set(rows)-consumed):add(name,[name])
 assert consumed==set(rows) and len(plans)==229
 return plans
def load_state(config,row):
 path=(config.artifact/row['file']).resolve();assert path.is_relative_to(config.artifact) and sha(path)==row['sha256']
 state=torch.load(path,map_location='cpu',weights_only=True)
 assert state['format']=='own_affine_dense4_v1' and state['group_size']==128 and state['source_dtype']=='torch.bfloat16'
 return state

class Native122DenseMethod(LinearMethodBase):
 def __init__(self,config,prefix):self.config=config;self.prefix=canonical_prefix(prefix)
 def create_weights(self,layer,input_size_per_partition,output_partition_sizes,input_size,output_size,params_dtype,**extra_weight_attrs):
  assert params_dtype==torch.bfloat16 and input_size_per_partition==input_size
  assert getattr(layer,'tp_size',1)==1 and not getattr(layer,'has_bias',False)
  rows=self.config.dense_plans[self.prefix]
  assert all(v['shape'][1]==input_size for v in rows) and sum(v['shape'][0] for v in rows)==output_size==sum(output_partition_sizes)
  parts=[]
  for row in rows:
   state=load_state(self.config,row);assert state['shape']==row['shape']
   parts.append(PackedAffine4NativeChunkedHead(state) if self.prefix=='lm_head' else PackedAffine4NativeLinear(state))
  layer.native122_parts=torch.nn.ModuleList(parts)
  assert not hasattr(layer,'weight') and all(v.is_cuda for v in layer.native122_parts.buffers())
 def process_weights_after_loading(self,layer):assert hasattr(layer,'native122_parts') and not hasattr(layer,'weight')
 def apply(self,layer,x,bias=None):
  assert bias is None and x.dtype==torch.bfloat16 and x.is_cuda
  outputs=[part(x) for part in layer.native122_parts]
  return outputs[0] if len(outputs)==1 else torch.cat(outputs,dim=-1)

class Native122EmbeddingMethod(QuantizeMethodBase):
 def __init__(self,config):self.config=config
 def create_weights(self,layer,input_size_per_partition,output_partition_sizes,input_size,output_size,params_dtype,**extra_weight_attrs):
  assert params_dtype==torch.bfloat16 and layer.tp_size==1 and input_size_per_partition==input_size==3072
  assert sum(output_partition_sizes)==output_size==layer.num_embeddings==248320
  state=load_state(self.config,self.config.embedding_entry);assert state['shape']==[248320,3072]
  layer.native122_embedding=PackedAffine4Embedding(state)
  assert not hasattr(layer,'weight') and all(v.is_cuda for v in layer.native122_embedding.buffers())
 def process_weights_after_loading(self,layer):assert hasattr(layer,'native122_embedding') and not hasattr(layer,'weight')
 def embedding(self,layer,input_):return layer.native122_embedding(input_.to(torch.int64))
 def apply(self,layer,x,bias=None):raise NotImplementedError('LM head uses the separately calibrated dense4 projection')

@register_quantization_config(NAME)
class Native122CompactConfig(Native122ExpertsConfig):
 def __init__(self,artifact):
  super().__init__(artifact);self.manifest=json.loads((self.artifact/'summary.json').read_text());self.dense_plans=dense_plan(self.manifest);self.embedding_entry=self.manifest['embedding']
  self.retained_names=set(self.manifest['retained_tensor_sha256'])
 @staticmethod
 def get_name():return NAME
 @staticmethod
 def get_config_filenames():return ['bonsai_native122_compact_control.json']
 @classmethod
 def from_config(cls,config):
  assert config.get('compact_model_control_only') is True and config.get('quant_method')==NAME
  return cls(config['artifact'])
 def get_quant_method(self,layer,prefix):
  canonical=canonical_prefix(prefix)
  if isinstance(layer,RoutedExperts):return Native122ExpertMethod(layer.moe_config,self,prefix)
  if isinstance(layer,ParallelLMHead):
   assert canonical=='lm_head';return Native122DenseMethod(self,prefix)
  if isinstance(layer,VocabParallelEmbedding):
   assert canonical=='embed_tokens';return Native122EmbeddingMethod(self)
  if isinstance(layer,LinearBase):
   if canonical in self.dense_plans:return Native122DenseMethod(self,prefix)
   assert 'model.language_model.'+canonical+'.weight' in self.retained_names,('Unmapped linear',prefix)
   return UnquantizedLinearMethod()
  # Attention quantization asks the same config but has no packed linear weights.
  return None
