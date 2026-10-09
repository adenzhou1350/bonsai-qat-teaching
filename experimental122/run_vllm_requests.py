"""Run the native4095 packed text model through pinned vLLM 0.24 on one5090.

Experimental serial resident requests. Supports an eager correctness control
and a full-decode-only CUDA Graph candidate; this is not an HTTP server.
"""
import argparse
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--model',type=Path,required=True,help='Existing native4095 packed model directory.')
 p.add_argument('--requests',type=Path,required=True,help='UTF-8 JSONL; prompt string in each row.')
 p.add_argument('--output',type=Path,required=True,help='New directory for result and weight proof.')
 p.add_argument('--decode-graph',action='store_true',help='Full decode graph, capture batch size1; prefill stays eager.')
 p.add_argument('--audit-graph-replays',action='store_true',help='Count actual graph replay calls; diagnostic timings only.')
 p.add_argument('--max-new-tokens',type=int,default=64)
 p.add_argument('--max-model-len',type=int,default=1536)
 p.add_argument('--kv-cache-mib',type=int,default=512,help='Explicit GPU KV/mamba cache budget, not model memory.')
 a=p.parse_args();assert 1<=a.max_new_tokens<=1024 and a.max_model_len>=128 and a.kv_cache_mib>=512
 assert not a.audit_graph_replays or a.decode_graph
 import gc,hashlib,json,os,sys,time
 os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0';os.environ['BONSAI_BATCHED_PREFILL']='1'
 package=Path(__file__).resolve().parent;sys.path.insert(0,str(package/'runtime'))
 from bonsai_native122_vllm_model_loader_v1593 import register_model,ARCHITECTURE,LOAD_FORMAT
 from bonsai_native122_vllm_dense_v1576 import NAME
 from bonsai_native122_vllm_experts_v1564 import sha
 import torch
 from transformers import AutoTokenizer
 from vllm import LLM,SamplingParams
 from vllm.compilation.cuda_graph import CUDAGraphWrapper
 assert torch.cuda.device_count()==1 and '5090' in torch.cuda.get_device_name();torch.set_num_threads(1)
 torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 provenance=package/'vllm-engine-provenance.json';policy=json.loads(provenance.read_text(encoding='utf-8'))['file_sha256']
 assert len(policy)==56 and all(sha(package.parent/n)==h for n,h in policy.items())
 output=a.output.resolve();output.mkdir(parents=True,exist_ok=False);(output/'source-policy.json').write_text(json.dumps(policy,indent=2))
 artifact=a.model.resolve();rows=[json.loads(s) for s in a.requests.read_text(encoding='utf-8').splitlines() if s.strip()]
 assert rows and all(isinstance(r,dict) and isinstance(r.get('prompt'),str) and r['prompt'].strip() for r in rows)
 tok=AutoTokenizer.from_pretrained(artifact,local_files_only=True)
 inputs=[tok.encode(tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False) for r in rows]
 # Only short prefill has been admitted by this entry; longer/chunk-boundary
 # numerical checks need their own evidence before this guard is expanded.
 assert max(map(len,inputs))<=128 and max(map(len,inputs))+a.max_new_tokens<a.max_model_len
 generation=json.loads((artifact/'generation_config.json').read_text());eos=generation['eos_token_id'];eos=set(eos if isinstance(eos,list) else [eos]);assert eos and None not in eos
 started=time.monotonic();report={'completed':False,'stage':'starting','artifact_manifest_sha256':sha(artifact/'summary.json'),'CPU_weight_offload':False,'eos_respected':True,'max_new_tokens':a.max_new_tokens,'max_model_len':a.max_model_len,'max_num_seqs':1,'KV_cache_budget_bytes':a.kv_cache_mib*1024**2,'decode_graph':a.decode_graph,'graph_replay_audit':a.audit_graph_replays,'requests':[],'speedup_claimed':False,'limits':'Experimental native4095 text-only actual vLLM0.24 single-GPU serial requests; maximum128 input tokens admitted here. Model load/tokenization/JSON excluded from request times; first inference can include JIT. Allocator peaks/request-end observations are not full NVML sampled peaks. Optional replay counters add instrumentation. Fixed authored examples are not capability or sustained-service evidence.'}
 def save(stage):
  report.update(stage=stage,wall_seconds=time.monotonic()-started);p=output/'result.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(report,ensure_ascii=False,indent=2));t.replace(p)
 replay_counter={'calls':0};original_replay=torch.cuda.CUDAGraph.replay
 def counted_replay(graph):
  replay_counter['calls']+=1;return original_replay(graph)
 if a.audit_graph_replays:torch.cuda.CUDAGraph.replay=counted_replay
 llm=None
 try:
  register_model();save('loading_full_vllm_engine')
  compilation={'mode':0,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[1]} if a.decode_graph else 0
  llm=LLM(model=str(artifact),tokenizer=str(artifact),dtype='bfloat16',quantization=NAME,load_format=LOAD_FORMAT,model_loader_extra_config={'weight_proof':str(output/'weight-proof.json')},hf_overrides={'architectures':[ARCHITECTURE],'quantization_config':{'quant_method':NAME,'artifact':str(artifact),'compact_model_control_only':True}},tensor_parallel_size=1,distributed_executor_backend='uni',max_num_seqs=1,max_model_len=a.max_model_len,max_num_batched_tokens=128,enforce_eager=not a.decode_graph,compilation_config=compilation,kv_cache_memory_bytes=a.kv_cache_mib*1024**2,enable_prefix_caching=False,gpu_memory_utilization=.97,cpu_offload_gb=0,seed=1594)
  report['actual_full_vllm_engine_constructed']=True;report['model_load_seconds']=time.monotonic()-started
  proof=json.loads((output/'weight-proof.json').read_text());assert proof['all361_retained_input_and_loaded_tensor_hashes_exact']
  report['unique_CUDA_model_storage_bytes']=proof['unique_CUDA_model_storage_bytes'];report['weight_proof_sha256']=sha(output/'weight-proof.json')
  entries=[{'mode':str(w.runtime_mode),'batch_descriptor':str(k),'actual_CUDAGraph_created':e.cudagraph is not None} for w in list(CUDAGraphWrapper._all_instances) for k,e in w.concrete_cudagraph_entries.items()]
  report['actual_capture_entries']=entries
  if a.decode_graph:assert entries and all(v['actual_CUDAGraph_created'] for v in entries),entries
  save('engine_loaded')
  for i,ids in enumerate(inputs):
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.monotonic();before=replay_counter['calls']
   outputs=llm.generate([{'prompt_token_ids':ids}],SamplingParams(temperature=0,max_tokens=a.max_new_tokens,ignore_eos=False),use_tqdm=False)
   torch.cuda.synchronize();elapsed=time.monotonic()-t;value=outputs[0].outputs[0];tokens=list(value.token_ids);assert 1<=len(tokens)<=a.max_new_tokens
   free,total=torch.cuda.mem_get_info();count=replay_counter['calls']-before
   if a.audit_graph_replays and len(tokens)>1:assert count>0
   report['requests'].append({'request_index':i,'input_tokens':len(ids),'output_tokens':len(tokens),'generated_token_ids':tokens,'text':value.text,'eos_reached':tokens[-1] in eos,'finish_reason':value.finish_reason,'request_seconds':elapsed,'end_to_end_output_tokens_per_second':len(tokens)/elapsed,'actual_CUDAGraph_replays':count if a.audit_graph_replays else None,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'device_used_bytes_at_observation':total-free,'device_total_bytes':total})
   save('completed_request_'+str(i));print(json.dumps({'request_index':i,'output_tokens':len(tokens),'request_seconds':elapsed,'actual_CUDAGraph_replays':count if a.audit_graph_replays else None}),flush=True)
  assert all(sha(package.parent/n)==h for n,h in policy.items());report['completed']=True;save('completed')
 finally:
  if llm is not None:llm.llm_engine.engine_core.shutdown();del llm;gc.collect()
  torch.cuda.CUDAGraph.replay=original_replay

if __name__=='__main__':main()
