"""Validate all native122 dense/embedding components through pinned vLLM 0.24.

Requires the admitted native4095 packed artifact and one RTX 5090.
This is a component numerical check, not a full model or throughput benchmark.
"""
import argparse
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--model',type=Path,required=True,help='Existing native4095 packed artifact.')
 p.add_argument('--output',type=Path,required=True,help='New directory for progress, source identity and final summary.')
 a=p.parse_args()
 import gc,hashlib,json,socket,sys,time
 root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);package=Path(__file__).resolve().parent;sys.path.insert(0,str(package/'runtime'))
 from bonsai_native122_vllm_dense_v1576 import Native122CompactConfig,Native122DenseMethod,Native122EmbeddingMethod,NAME,PINNED_ADDITIONAL_SOURCE
 from bonsai_native122_vllm_experts_v1564 import sha
 import torch
 from torch.nn import functional as F
 from gemq_dense_runtime import backend
 from vllm.config import VllmConfig,set_current_vllm_config
 from vllm.distributed import init_distributed_environment,initialize_model_parallel,destroy_model_parallel,destroy_distributed_environment
 from vllm.model_executor.layers.linear import ReplicatedLinear,RowParallelLinear,MergedColumnParallelLinear,QKVParallelLinear
 from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding,ParallelLMHead
 from vllm.model_executor.layers.logits_processor import LogitsProcessor
 policy=json.loads((package/'vllm-components-provenance.json').read_text(encoding='utf-8'))['file_sha256'];assert len(policy)==54 and all(sha(package.parent/n)==h for n,h in policy.items());(root/'source-policy.json').write_text(json.dumps(policy,indent=2))
 assert torch.cuda.device_count()==1 and '5090' in torch.cuda.get_device_name();torch.set_num_threads(1);backend();start=time.monotonic()
 artifact=a.model.resolve()
 config=Native122CompactConfig.from_config({'quant_method':NAME,'artifact':str(artifact),'compact_model_control_only':True})
 out={'passed':False,'stage':'starting','source_policy_sha256':sha(root/'source-policy.json'),'actual_vllm_version':'0.24.0','pinned_additional_source_SHA256':PINNED_ADDITIONAL_SOURCE,'artifact_manifest_sha256':sha(artifact/'summary.json'),'banks':[],'cases':[],'embedding_cases':[],'actual_dense_module_constructions':0,'independent_relative_L2_tolerance':.005,'whole_vllm_engine_run':False,'speedup_claimed':False,'quality_accepted':False,'reserved_inputs_opened':False}
 def save(stage):
  out.update(stage=stage,seconds=time.monotonic()-start);p=root/'progress.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(out,indent=2));t.replace(p)
 def unpack_cpu(state):
  # BF16 subtraction rounds before BF16 multiplication, as in the artifact contract.
  codes=state['codes'];n,k=state['shape'];values=torch.stack((codes&15,codes>>4),dim=-1).reshape(n,k).bfloat16()
  zero=state['zeros'].repeat_interleave(128,-1);scale=state['scales'].repeat_interleave(128,-1)
  return ((values-zero).bfloat16()*scale).bfloat16()
 def independent_weight(state):
  n,k=state['shape'];reference=torch.empty((n,k),device='cuda',dtype=torch.bfloat16)
  for first in range(0,n,2048):
   end=min(first+2048,n);chunk={**state,'shape':[end-first,k],'codes':state['codes'][first:end],'zeros':state['zeros'][first:end],'scales':state['scales'][first:end]}
   reference[first:end].copy_(unpack_cpu(chunk).cuda())
  return reference
 def errors(observed,expected):
  assert observed.shape==expected.shape and torch.isfinite(observed).all() and torch.isfinite(expected).all()
  return {'bitwise_equal':torch.equal(observed,expected),'relative_L2':float((observed.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-12)),'maximum_absolute_error':float((observed.float()-expected.float()).abs().max())}
 def construct(prefix,rows):
  k=rows[0]['shape'][1];n=sum(v['shape'][0] for v in rows);common={'bias':False,'params_dtype':torch.bfloat16,'quant_config':config,'prefix':('lm_head' if prefix=='lm_head' else 'model.'+prefix)}
  if prefix=='lm_head':return ParallelLMHead(n,k,**common)
  common.update(return_bias=False,disable_tp=True)
  if prefix.endswith('.self_attn.qkv_proj'):return QKVParallelLinear(k,256,64,2,**common)
  if prefix.endswith('.linear_attn.in_proj_qkvz'):return MergedColumnParallelLinear(k,[2048,2048,8192,8192],**common)
  if prefix.endswith('.linear_attn.in_proj_ba'):return MergedColumnParallelLinear(k,[64,64],**common)
  if prefix.endswith('.mlp.shared_expert.gate_up_proj'):return MergedColumnParallelLinear(k,[1024,1024],**common)
  if prefix.endswith(('.out_proj','.o_proj','.down_proj')):return RowParallelLinear(k,n,**common)
  return ReplicatedLinear(k,n,**common)
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 with set_current_vllm_config(VllmConfig()):
  init_distributed_environment(world_size=1,rank=0,local_rank=0,distributed_init_method=f'tcp://127.0.0.1:{port}',backend='gloo');initialize_model_parallel(backend='gloo')
  try:
   with torch.inference_mode():
    for index,(prefix,rows) in enumerate(config.dense_plans.items()):
     save('constructing_'+prefix);actual=construct(prefix,rows);assert isinstance(actual.quant_method,Native122DenseMethod);actual.quant_method.process_weights_after_loading(actual);actual.eval();out['actual_dense_module_constructions']+=1;refs=[]
     assert not hasattr(actual,'weight') and not list(actual.parameters()) and all(v.is_cuda for v in actual.buffers())
     for row,part in zip(rows,actual.native122_parts):
      assert sha(artifact/row['file'])==row['sha256'];state=torch.load(artifact/row['file'],map_location='cpu',weights_only=True);reference=independent_weight(state);decoded=part.dequant();assert torch.equal(decoded,reference),(prefix,row['module'])
      out['banks'].append({'source_module':row['module'],'bank_sha256':row['sha256'],'shape':row['shape'],'all_weight_elements_match_independent_CPU_unpack_bitwise':True});refs.append(reference);del decoded,state,reference
     generator=torch.Generator(device='cuda').manual_seed(15790+index);k=rows[0]['shape'][1]
     for m in (1,4,64):
      x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16,generator=generator)*.125;outputs=[]
      for reference in refs:
       if prefix=='lm_head':
        y=torch.empty((m,len(reference)),device='cuda',dtype=torch.bfloat16)
        for first in range(0,len(reference),2048):y[:,first:first+2048]=F.linear(x,reference[first:first+2048])
       else:y=F.linear(x,reference)
       outputs.append(y)
      expected=outputs[0] if len(outputs)==1 else torch.cat(outputs,-1)
      observed=LogitsProcessor(248320)._get_logits(x,actual,None) if prefix=='lm_head' else actual(x)
      error=errors(observed,expected);item={'target_module':prefix,'actual_vllm_class':type(actual).__name__,'tokens':m,'source_projection_boundaries_preserved':True,**error};item['passed']=error['relative_L2']<.005;out['cases'].append(item);save('checked_dense_case');assert item['passed'],item;del x,outputs,expected,observed,y
     del actual,refs,reference;gc.collect();torch.cuda.empty_cache();save('completed_dense_module');print(json.dumps({'dense_modules':index+1,'banks':len(out['banks']),'cases':len(out['cases'])}),flush=True)
    entry=config.embedding_entry;assert sha(artifact/entry['file'])==entry['sha256'];state=torch.load(artifact/entry['file'],map_location='cpu',weights_only=True)
    embedding=VocabParallelEmbedding(248320,3072,params_dtype=torch.bfloat16,quant_config=config,prefix='model.embed_tokens');assert isinstance(embedding.quant_method,Native122EmbeddingMethod);embedding.quant_method.process_weights_after_loading(embedding)
    assert not hasattr(embedding,'weight') and not list(embedding.parameters()) and all(v.is_cuda for v in embedding.buffers())
    reference=independent_weight(state);ids=torch.arange(248320,device='cuda');decoded=embedding.native122_embedding.base(ids);assert torch.equal(decoded,reference)
    out['all248320_embedding_rows_base_match_independent_CPU_unpack_bitwise']=True;del ids,decoded
    a=state['embedding_A'].cuda();b=state['embedding_B'].cuda();assert state['lora_scale']==.125
    for label,ids in [('single',torch.tensor([248044],device='cuda')),('boundaries',torch.tensor([0,248043,248044,248319],device='cuda')),('duplicates',torch.tensor([0,0,248319,248319],device='cuda')),('matrix2x4',torch.tensor([[0,1,128,129],[248042,248043,248044,248319]],device='cuda'))]:
     expected=reference[ids]+F.linear(a[ids],b)*.125;observed=embedding(ids);error=errors(observed,expected);item={'kind':label,'shape':list(ids.shape),'IDs':ids.cpu().tolist(),'actual_VocabParallelEmbedding_forward_executed':True,**error};item['passed']=error['relative_L2']<.005;out['embedding_cases'].append(item);save('checked_embedding_case');assert item['passed'],item;del ids,expected,observed
    out['embedding_bank_sha256']=entry['sha256'];del state,embedding,reference,a,b;gc.collect();torch.cuda.empty_cache()
  finally:destroy_model_parallel();destroy_distributed_environment()
 assert len(out['banks'])==373 and len(out['cases'])==687 and len(out['embedding_cases'])==4 and out['actual_dense_module_constructions']==229
 assert all(sha(package.parent/n)==h for n,h in policy.items())
 out.update(passed=True,diagnostic_allocator_peak_bytes=torch.cuda.max_memory_allocated(),limits='Real vLLM0.24 TP1 Replicated/Row/Merged/QKV linears and ParallelLMHead/LogitsProcessor dispatch, all373 dense4 banks/229 projection modules; full-weight independent CPU BF16 dequant checks,1/4/64token synthetic independent F.linear checks. Original source projection and head2048row rounding boundaries retained. Real VocabParallelEmbedding forward, all248320base rows independently checked, rank16 corrections at boundary/duplicate/matrix IDs. RelativeL2 threshold unchanged .005; diagnostic arrays not serialized. Isolated components, not full-model loader/attention/cache/generation/continuous batching/quality/throughput or deployment memory proof.')
 save('completed');(root/'summary.json').write_text(json.dumps(out,indent=2));print(json.dumps({'passed':True,'dense_banks':373,'embedding_cases':4}))

if __name__=='__main__':main()
