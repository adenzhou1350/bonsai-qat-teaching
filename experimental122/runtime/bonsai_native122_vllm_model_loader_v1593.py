"""Text-only native4095 packed model loader for pinned vLLM 0.24.

Experimental full-engine integration. Component proofs do not prove attention,
cache, generation equality or throughput of this wrapper.
"""
import hashlib,importlib.util,json
from pathlib import Path
import torch
from bonsai_native122_vllm_dense_v1576 import Native122CompactConfig,Native122EmbeddingMethod
from bonsai_native122_vllm_experts_v1564 import sha
from vllm import ModelRegistry
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.models import qwen3_5
from vllm.model_executor.models.config import MODELS_CONFIG_MAP,Qwen3_5ForConditionalGenerationConfig
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

ARCHITECTURE='BonsaiNative122PackedTextForCausalLM'
LOAD_FORMAT='bonsai_native122_packed_v1'
PINNED_LOADER_SOURCE={
 'model_executor/model_loader/__init__.py':'139a92f930f3e508836932ea6eaacd63d9d4e9835f9bf545c59e84fce08a8aed',
 'model_executor/model_loader/base_loader.py':'0632e9db1988d77db65e1b5ab5eb79e9254d8191ab45e4d772573d5cf3649aa1',
 'model_executor/models/config.py':'6f812bebdd0ffdd82bfe237971efd6614cfa22197e345c63e86a5564bc6f1333',
}
base=Path(importlib.util.find_spec('vllm').origin).parent
assert all(sha(base/n)==h for n,h in PINNED_LOADER_SOURCE.items())

def tensor_sha(value):
 raw=value.detach().contiguous().view(torch.uint8).reshape(-1);h=hashlib.sha256()
 for first in range(0,len(raw),16*1024*1024):h.update(raw[first:first+16*1024*1024].cpu().numpy().tobytes())
 return h.hexdigest()

def retained_name(name):
 assert name.startswith('model.language_model.'),name
 return 'model.'+name[len('model.language_model.'):]

class BonsaiNative122PackedTextForCausalLM(qwen3_5.Qwen3_5MoeForCausalLM):
 is_hybrid=True
 supports_mrope=True
 def get_mrope_input_positions(self,input_tokens,mm_features):
  assert not mm_features,'Text-only packed model'
  return Qwen3VLForConditionalGeneration._get_mrope_input_positions(input_tokens=input_tokens,mm_features=mm_features,config=self.bonsai_original_hf_config)
 @classmethod
 def get_mamba_state_dtype_from_config(cls,vllm_config):return qwen3_5.Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config(vllm_config)
 @classmethod
 def get_mamba_state_shape_from_config(cls,vllm_config):return qwen3_5.Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config(vllm_config)
 @classmethod
 def get_mamba_state_copy_func(cls):return qwen3_5.Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func()
 def __init__(self,*,vllm_config,prefix=''):
  config=vllm_config.quant_config;assert isinstance(config,Native122CompactConfig)
  parallel=vllm_config.parallel_config
  assert parallel.tensor_parallel_size==parallel.pipeline_parallel_size==1 and not parallel.enable_eplb
  assert vllm_config.model_config.is_hybrid and vllm_config.cache_config.mamba_block_size is not None
  self.bonsai_original_hf_config=vllm_config.model_config.hf_config
  # Pinned Qwen3_5Model creates embedding without forwarding quant_config/prefix.
  # Supply those constructor arguments only while this text model is created;
  # never allocate the original full BF16 vocabulary matrix first.
  original=qwen3_5.VocabParallelEmbedding;assert original is VocabParallelEmbedding
  class CompactEmbedding(VocabParallelEmbedding):
   def __init__(self,*args,**kwargs):
    assert kwargs.get('quant_config',config) is config
    kwargs.update(quant_config=config,prefix='model.embed_tokens',params_dtype=torch.bfloat16)
    super().__init__(*args,**kwargs)
  qwen3_5.VocabParallelEmbedding=CompactEmbedding
  try:super().__init__(vllm_config=vllm_config,prefix=prefix)
  finally:qwen3_5.VocabParallelEmbedding=original
  assert isinstance(self.model.embed_tokens.quant_method,Native122EmbeddingMethod)
  assert not hasattr(self.model.embed_tokens,'weight')

@register_model_loader(LOAD_FORMAT)
class Native122PackedModelLoader(BaseModelLoader):
 def download_model(self,model_config):
  assert Path(model_config.model).is_dir(),'Existing local packed model required'
 def load_weights(self,model,model_config):
  assert isinstance(model,BonsaiNative122PackedTextForCausalLM)
  config=model.quant_config;artifact=config.artifact;manifest=config.manifest
  retained_file=artifact/'retained.pt';assert sha(retained_file)==manifest['retained_sha256']
  values=torch.load(retained_file,map_location='cpu',weights_only=True)
  expected=manifest['retained_tensor_sha256'];assert set(values)==set(expected) and len(values)==361
  params=dict(model.named_parameters());mapped={retained_name(n):n for n in values}
  assert len(mapped)==361 and set(params)==set(mapped),{'missing':sorted(set(mapped)-set(params)),'unexpected':sorted(set(params)-set(mapped))}
  for name,value in values.items():
   assert tensor_sha(value)==expected[name],name
   target=params[retained_name(name)];assert target.shape==value.shape and target.is_cuda,name
   # Preserve Parameter identity and attached weight_loader; retain FP32 GDN
   # norm/A_log rather than silently casting the original tensors to BF16.
   if target.dtype!=value.dtype:target.data=target.data.to(value.dtype)
  loaded=model.load_weights((retained_name(n),v) for n,v in values.items())
  assert set(loaded)==set(mapped),{'loaded':len(loaded),'expected':len(mapped)}
  post={}
  for target,source in mapped.items():
   actual=params[target];assert actual.dtype==values[source].dtype and tensor_sha(actual)==expected[source],source
   post[source]=expected[source]
  del values
  experts=[m for m in model.modules() if isinstance(m,RoutedExperts)]
  assert len(experts)==48 and all(hasattr(m,'bonsai_experts') and not hasattr(m,'w13_weight') and not hasattr(m,'w2_weight') for m in experts)
  dense=[m for m in model.modules() if hasattr(m,'native122_parts')]
  assert len(dense)==229 and sum(len(m.native122_parts) for m in dense)==373
  assert not hasattr(model.lm_head,'weight') and not hasattr(model.model.embed_tokens,'weight')
  tensors=[v for _,v in [*model.named_parameters(),*model.named_buffers()]];assert all(v.is_cuda for v in tensors)
  stores={v.untyped_storage().data_ptr():v.untyped_storage().nbytes() for v in tensors}
  proof={'all361_retained_input_and_loaded_tensor_hashes_exact':True,'actual_compact_RoutedExperts_layers':48,'actual_dense_projection_modules':229,'source_dense_banks':373,'actual_embedding_quant_method':type(model.model.embed_tokens.quant_method).__name__,'embedding_constructor_quant_config_supplied':True,'temporary_constructor_override_restored':qwen3_5.VocabParallelEmbedding is VocabParallelEmbedding,'retained_postload_tensor_sha256':post,'all_named_model_weights_and_buffers_CUDA':True,'unique_CUDA_model_storage_bytes':sum(stores.values()),'CPU_weight_offload':False,'persistent_BF16_expert_or_head_or_embedding_weight':False,'pinned_loader_source_sha256':PINNED_LOADER_SOURCE,'whole_engine_generation_validated':False,'speedup_claimed':False}
  extra=self.load_config.model_loader_extra_config;assert set(extra)=={'weight_proof'}
  output=Path(extra['weight_proof']);assert output.parent.is_dir() and not output.exists();output.write_text(json.dumps(proof,indent=2))

def register_model():
 MODELS_CONFIG_MAP[ARCHITECTURE]=Qwen3_5ForConditionalGenerationConfig
 ModelRegistry.register_model(ARCHITECTURE,BonsaiNative122PackedTextForCausalLM)
