"""Inference dependency extracted from validated experimental runtime."""

from native_moe_shared_defer_v2_dead_router import bind_moe_shared_defer_model

from linear_attention_prefill_lifetime_v6_detached_qkv import bind_linear_prefill_lifetime_model

from bounded_attention_rms_norm_v1 import bind_bounded_attention_rms_model

from bounded_head_sdpa_v6_output_gate import bind_bounded_head_sdpa_model

from bounded_gated_norm_v1 import bind_gated_norm_model

import gc

from carry_fp32_segmented_delta_prefill_v5_preallocated import bind_segmented_model

from pathlib import Path

import torch

from transformers import AutoConfig, AutoTokenizer, GenerationConfig, Qwen3_5MoeForConditionalGeneration

from gemq_dense_runtime import backend, sha

from qtea_source_expert_probe import tensor_sha

from primary_gsq122_dense4_runtime_v1 import bank_rows, PackedAffine4NativeChunkedHead

from gsq_ternary_retained_factor_runtime_v18_expert_stream import PackedGroupedRetainedFactorTritExperts as PackedRetainedFactorTritExperts

from affine4_packed_native import PackedAffine4NativeLinear

from embedding4_packed_runtime import PackedAffine4Embedding

FORMAT='research-direct-ternary122-dense4-original-teacher-rank8-recovery-v1'

def read_banks(directory,rows):
 banks={}
 for name,row in rows.items():
  bank=torch.load(Path(directory)/row['file'],map_location='cpu',weights_only=True)
  assert bank['format']=='own_gsq_five_trits_expert_bank_v1' and bank['origin']=='direct_lloyd8_source_ptq_not_official_gsq' and bank['shape']==row['shape'] and len(bank['states'])==256
  assert all(v['exact_source_bf16_grid'] and v['scales'].dtype==torch.bfloat16 for v in bank['states'])
  factor=bank['retained_factor'];assert factor['format']=='retained_rank8_bf16_factors_v1' and factor['rank']==8 and factor['scale']==1.
  assert factor['a'].dtype==factor['b'].dtype==torch.bfloat16
  banks[name]=bank
 return banks

def check_retained(retained,report):
 assert sum(v.numel()*v.element_size() for v in retained.values())==report['retained_tensor_bytes']
 assert set(retained)==set(report['retained_tensor_sha256'])
 for name,value in retained.items():assert tensor_sha(value)==report['retained_tensor_sha256'][name]

def assemble_controlled(directory,report):
 directory=Path(directory);backend();cfg=AutoConfig.from_pretrained(directory,local_files_only=True);cfg._attn_implementation='sdpa';cfg.text_config._attn_implementation='sdpa';cfg.text_config._experts_implementation='grouped_mm';assert cfg.text_config.num_hidden_layers==48 and cfg.text_config.hidden_size==3072
 with torch.device('meta'):model=Qwen3_5MoeForConditionalGeneration(cfg)
 language=model.model.language_model;rows,dense=bank_rows(directory,report);retained=torch.load(directory/'retained.pt',map_location='cpu',weights_only=True);check_retained(retained,report)
 for li in range(48):
  banks=read_banks(directory,rows[li]);language.layers[li].mlp.experts=PackedRetainedFactorTritExperts(banks);del banks;gc.collect()
 for row in dense:
  state=torch.load(directory/row['file'],map_location='cpu',weights_only=True);assert state['shape']==row['shape']
  if row['module']=='lm_head':model.lm_head=PackedAffine4NativeChunkedHead(state)
  else:
   parent,name=row['module'].rsplit('.',1);setattr(language.get_submodule(parent),name,PackedAffine4NativeLinear(state,retained.get('model.language_model.'+row['module']+'.bias')))
  del state
 ep=directory/report['embedding']['file'];assert sha(ep)==report['embedding']['sha256'];language.embed_tokens=PackedAffine4Embedding(torch.load(ep,map_location='cpu',weights_only=True))
 for name,value in retained.items():
  parent,attr=name.rsplit('.',1);module=model.get_submodule(parent)
  if attr in module._parameters:module._parameters[attr]=torch.nn.Parameter(value.to('cuda'),requires_grad=False)
  elif attr in module._buffers:module._buffers[attr]=value.to('cuda')
  else:raise AssertionError(name)
 language.rotary_emb=type(language.rotary_emb)(cfg.text_config,device=torch.device('cuda'));model.requires_grad_(False);model.eval()
 for name,value in [*language.named_parameters(),*language.named_buffers(),*model.lm_head.named_parameters(),*model.lm_head.named_buffers()]:assert value.device.type=='cuda',name
 model.generation_config=GenerationConfig.from_pretrained(directory,local_files_only=True);del retained;gc.collect()
 bind_segmented_model(model);bind_gated_norm_model(model);bind_bounded_head_sdpa_model(model);bind_bounded_attention_rms_model(model);bind_linear_prefill_lifetime_model(model);bind_moe_shared_defer_model(model)
 return AutoTokenizer.from_pretrained(directory,local_files_only=True),model,report
