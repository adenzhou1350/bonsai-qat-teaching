"""Pinned vLLM 0.24 expert-only adapter for the native4095 122B artifact.

This does not load dense/embedding quantization or run a vLLM engine.
MoERunner owns shared-expert execution and addition; apply returns routed output.
"""
import hashlib,importlib.metadata,importlib.util,json,re
from pathlib import Path

MANIFEST_SHA256='dc3a9ba1a8bd9bcdd229b3c259cc5d676719861946a5ed2d90b39ac2e196def1'
NAME='bonsai_native122_experts_control_v1'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while block:=f.read(8*1024*1024):h.update(block)
 return h.hexdigest()
def pinned_contract():
 assert importlib.metadata.version('vllm')=='0.24.0'
 base=Path(importlib.util.find_spec('vllm').origin).parent
 expected={
 'model_executor/layers/fused_moe/fused_moe_method_base.py':'b8d3f53f46614f2db1b4cf9be058296f84191e2437887fb0dd3dfc42bd8c5f48',
 'model_executor/layers/fused_moe/routed_experts.py':'957e62a0acbe665f5a15641161f2854d982ebdbffb74456b405105533ed8ed05',
 'model_executor/layers/quantization/base_config.py':'260c70dbb390308fae32b07e09d275ad6003c9125bcdbaa6eb477e8a815ab7fe',
 'model_executor/layers/quantization/__init__.py':'00b8bb4e90e334efb978d97395072ad6d4c64754cd55859c33cb580ff011cb7a',
 'model_executor/layers/fused_moe/runner/moe_runner.py':'d11a841e24f46a14884d36334b83387370cd8232679d8569a6a06eb3a6918eb2',
 }
 assert all(sha(base/n)==v for n,v in expected.items()),'Installed vLLM source differs from inspected contract'
 return expected
PINNED_SOURCE=pinned_contract()
import torch
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts,SharedExpertsOrder
from gsq_ternary_retained_factor_runtime_v18_expert_stream import PackedGroupedRetainedFactorTritExperts

class Native122ExpertMethod(FusedMoEMethodBase):
 def __init__(self,moe,config,prefix):super().__init__(moe);self.config=config;self.prefix=prefix
 def maybe_roundup_sizes(self,hidden_size,intermediate_size_per_partition,act_dtype,moe_parallel_config):
  assert moe_parallel_config.tp_size==moe_parallel_config.ep_size==moe_parallel_config.dp_size==moe_parallel_config.pcp_size==moe_parallel_config.sp_size==1 and not moe_parallel_config.enable_eplb
  assert (hidden_size,intermediate_size_per_partition,act_dtype)==(3072,1024,torch.bfloat16)
  return hidden_size,intermediate_size_per_partition
 def create_weights(self,layer,num_experts,hidden_size,intermediate_size_per_partition,params_dtype,**extra_weight_attrs):
  assert (num_experts,hidden_size,intermediate_size_per_partition)==(256,3072,1024) and params_dtype==torch.bfloat16
  assert self.moe.activation.value=='silu' and not self.moe.has_bias
  match=re.search(r'(?:^|\.)layers\.(\d+)\.mlp\.experts$',self.prefix);assert match,self.prefix
  index=int(match[1]);assert 0<=index<48;banks={}
  for attr in ('gate_up_proj','down_proj'):
   entry=self.config.entries[(index,attr)];path=(self.config.artifact/entry['file']).resolve();assert path.is_relative_to(self.config.artifact) and sha(path)==entry['sha256']
   banks[attr]=torch.load(path,map_location='cpu',weights_only=True)
  layer.bonsai_experts=PackedGroupedRetainedFactorTritExperts(banks)
  assert not list(layer.bonsai_experts.parameters()) and all(v.is_cuda for v in layer.bonsai_experts.buffers())
  layer.bonsai_layer_index=index;layer.bonsai_apply_calls=0
 def process_weights_after_loading(self,layer):
  assert hasattr(layer,'bonsai_experts') and not hasattr(layer,'w13_weight') and not hasattr(layer,'w2_weight')
 def get_fused_moe_quant_config(self,layer):return FusedMoEQuantConfig.make()
 def apply(self,layer,x,topk_weights,topk_ids,shared_experts,shared_experts_input):
  if shared_experts is not None:
   assert isinstance(shared_experts,SharedExperts) and shared_experts_input is not None and not self.mk_can_overlap_shared_experts
   assert shared_experts._determine_shared_experts_order(shared_experts_input) in (SharedExpertsOrder.NO_OVERLAP,SharedExpertsOrder.MULTI_STREAM_OVERLAPPED)
  assert x.ndim==2 and x.shape[1]==3072 and x.dtype==torch.bfloat16 and x.is_cuda
  assert topk_ids.shape==topk_weights.shape==(len(x),8)
  layer.bonsai_apply_calls+=1
  return layer.bonsai_experts(x,topk_ids.to(torch.int64),topk_weights.to(x.dtype))

@register_quantization_config(NAME)
class Native122ExpertsConfig(QuantizationConfig):
 def __init__(self,artifact):
  super().__init__();self.artifact=Path(artifact).resolve();assert sha(self.artifact/'summary.json')==MANIFEST_SHA256
  report=json.loads((self.artifact/'summary.json').read_text());assert report['passed'] and report['full_model_exported'] and report['layers']==list(range(48)) and report['actual_optimizer_updates']==4096 and report['teacher_input_positions']==4095
  self.entries={}
  for row in report['matrices']:
   match=re.fullmatch(r'layers\.(\d+)\.mlp\.experts',row['module'])
   if match:
    assert row['packing_format']=='own-gsq-five-trits-expert-bank-v1'
    key=(int(match[1]),row['attr']);assert key not in self.entries;self.entries[key]=row
  assert set(self.entries)=={(li,attr) for li in range(48) for attr in ('gate_up_proj','down_proj')}
 @staticmethod
 def get_name():return NAME
 @staticmethod
 def get_supported_act_dtypes():return [torch.bfloat16]
 @classmethod
 def get_min_capability(cls):return 120
 @staticmethod
 def get_config_filenames():return ['bonsai_native122_experts_control.json']
 @classmethod
 def from_config(cls,config):
  assert config.get('expert_component_control_only') is True and config.get('quant_method')==NAME
  return cls(config['artifact'])
 def get_quant_method(self,layer,prefix):
  if isinstance(layer,RoutedExperts):return Native122ExpertMethod(layer.moe_config,self,prefix)
  raise NotImplementedError('Expert-only control: dense, embedding, full loader and engine are not implemented')
