"""Deployment text-window cache: verify the previously proved GPU prefix hash.

Does not allocate the original full rotary cache; window1536, text only.
"""
import copy,importlib.util,json
from pathlib import Path
import torch
from bonsai_native122_vllm_model_loader_v1593 import Native122PackedModelLoader,ARCHITECTURE,register_model,tensor_sha
from bonsai_native122_vllm_experts_v1564 import sha
from bonsai_native122_vllm_dense_v1576 import NAME
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

LOAD_FORMAT='bonsai_native122_text_window_packed_v1'
PINNED_ROTARY_SOURCE={
 'model_executor/layers/rotary_embedding/base.py':'c81804498f08f072356a26b41967f211475df2694dabf5da9f488c074d1fdef5',
 'model_executor/layers/rotary_embedding/mrope.py':'d5fc5ca640bf47fdfc27447bcc6b67b2294c9ac840bbc736fe75cc6098e2a51c',
 'model_executor/layers/rotary_embedding/__init__.py':'f703d3f7ca6be6b41837d4a07a27b94d1442149554bb4f4837cd3722671a5bbd',
}
base=Path(importlib.util.find_spec('vllm').origin).parent
assert all(sha(base/n)==h for n,h in PINNED_ROTARY_SOURCE.items())

def text_window_overrides(config,artifact):
 config=copy.deepcopy(config)
 if not hasattr(config,'text_config'):
  assert config.model_type=='dummy_qwen3_5_moe'
  return config  # vLLM parser first probes a bare PreTrainedConfig
 text=config.text_config
 assert text.max_position_embeddings==262144 and text.rope_parameters['rope_type']=='default' and text.rope_parameters['rope_theta']==10000000
 assert text.rope_parameters['mrope_section']==[11,11,10] and text.rope_parameters['partial_rotary_factor']==.25
 config.architectures=[ARCHITECTURE];config.quantization_config={'quant_method':NAME,'artifact':str(artifact),'compact_model_control_only':True}
 text.max_position_embeddings=1536
 return config

@register_model_loader(LOAD_FORMAT)
class Native122TextWindowPackedModelLoader(Native122PackedModelLoader):
 def load_weights(self,model,model_config):
  super().load_weights(model,model_config)
  assert model.config.max_position_embeddings==model_config.max_model_len==1536
  modules={id(layer.self_attn.rotary_emb):layer.self_attn.rotary_emb for layer in model.model.layers if hasattr(layer,'self_attn')};assert len(modules)==1
  rotary=next(iter(modules.values()));short=rotary.cos_sin_cache;assert short.shape==(6144,64) and short.dtype==torch.bfloat16 and short.is_cuda
  actual_hash=tensor_sha(short)
  assert actual_hash=='e8e4b56df2f9917ccdbd96f509cfb3e905ed5dd81c8c000dc4897d65fdb3fe58'
  prefix={'actual_GPU_candidate_cache_SHA256':actual_hash,'candidate_cache_bytes':short.numel()*short.element_size(),'original_cache_bytes_in_independent_GPU_control':134217728,'reclaimed_static_cache_bytes':133431296,'full_reference_GPU_cache_allocated_by_this_loader':False,'matches_independently_proved_GPU_prefix_hash':True,'independent_GPU_control_summary_SHA256':'174a057882d3e187dd01717c48119ba23fcb3cc78ea91e138d5d2642c43be8bc','theta_dimensions_dtype_and_mrope_layout_unchanged':True,'source_SHA256':PINNED_ROTARY_SOURCE}
  p=Path(self.load_config.model_loader_extra_config['weight_proof']);v=json.loads(p.read_text());v['text_window_rotary_cache']=prefix;p.write_text(json.dumps(v,indent=2));torch.cuda.empty_cache()
