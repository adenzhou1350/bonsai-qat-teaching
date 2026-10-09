"""Keep the packed122 model resident and reuse one single-request CUDA Graph."""
import argparse,gc,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
from infer import manifest,ARTIFACTS

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--model',required=True,type=Path)
 p.add_argument('--artifact-version',choices=('native4095',),default='native4095')
 p.add_argument('--requests',required=True,type=Path,help='UTF-8 JSONL; each row has a prompt string.')
 p.add_argument('--output',required=True,type=Path)
 p.add_argument('--max-new-tokens',type=int,default=96)
 p.add_argument('--max-cache-len',type=int,default=1536)
 p.add_argument('--auto-cache-len',action='store_true',help='Select the smallest cache bucket for each request; keep only one graph resident.')
 p.add_argument('--trim-prefill-cache',action='store_true',help='Release unused allocator blocks after prefill, preserving live graph/cache storage.')
 a=p.parse_args();assert 1<=a.max_new_tokens<=1024 and 128<=a.max_cache_len<=16384
 assert not a.output.exists(),'Choose a new output file; previous results are preserved.'
 rows=[json.loads(line) for line in a.requests.read_text(encoding='utf-8').splitlines() if line.strip()]
 assert rows and all(isinstance(r,dict) and isinstance(r.get('prompt'),str) and r['prompt'].strip() for r in rows)
 os.environ['BONSAI_BATCHED_PREFILL']='1'
 import torch
 from direct122_original_recovery_runtime_v38_expert_stream import assemble_controlled
 from reusable122_graph_decoder_v1 import ReusableGraphDecoder
 assert torch.cuda.device_count()==1 and '5090' in torch.cuda.get_device_name(0)
 torch.set_num_threads(1);loaded=time.perf_counter();directory,report=manifest(a.model,a.artifact_version)
 tok,model,_=assemble_controlled(directory,report)
 from lazy122_long_causal_mask_v2 import bind_lazy_long_mask_model,STATS
 bind_lazy_long_mask_model(model)
 tensors=[v for n,v in [*model.named_parameters(),*model.named_buffers()] if n.startswith(('model.language_model.','lm_head.'))]
 assert tensors and all(v.is_cuda for v in tensors)
 stores={v.untyped_storage().data_ptr():v.untyped_storage().nbytes() for v in tensors}
 eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);assert eos and None not in eos
 result={'model':'Qwen3.5-122B-A10B','artifact_manifest_sha256':ARTIFACTS[a.artifact_version][0],'model_load_seconds':time.perf_counter()-loaded,'unique_CUDA_text_storage_bytes':sum(stores.values()),'CPU_weight_offload':False,'batch_size':1,'max_cache_len':a.max_cache_len,'auto_cache_len':a.auto_cache_len,'trim_prefill_cache':a.trim_prefill_cache,'max_new_tokens':a.max_new_tokens,'eos_respected':True,'requests':[],'completed':False,'limits':'Resident single worker, serial requests, greedy generation; no concurrent batching, streaming API or vLLM. Request timing excludes model load, includes prefill and first graph capture. Graph reuse was verified on short prompts at max_cache_len1536; other capacities require separate validation. Original artifact quality scores remain below BF16.'}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 def save():
  f=a.output.with_suffix(a.output.suffix+'.tmp');f.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');f.replace(a.output)
 decoder=None;captures=0;reuses=0
 with torch.inference_mode():
  for index,row in enumerate(rows):
   text=tok.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   ids=torch.tensor(tok.encode(text,add_special_tokens=False),device='cuda')[None]
   assert ids.shape[1]+a.max_new_tokens+4<a.max_cache_len,('Request exceeds fixed cache capacity',index)
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
   needed=ids.shape[1]+a.max_new_tokens+4
   capacity=min(v for v in sorted({a.max_cache_len,1536,4096,8192,16384}) if needed<v<=a.max_cache_len) if a.auto_cache_len else a.max_cache_len
   capacity_changed=decoder is not None and decoder.max_cache_len!=capacity
   release_for_long_prefill=decoder is not None and a.auto_cache_len and ids.shape[1]>4096
   changed=capacity_changed or release_for_long_prefill
   if changed:
    decoder.graph.reset();del decoder;decoder=None;gc.collect();torch.cuda.empty_cache()
   if decoder is None:
    decoder=ReusableGraphDecoder(model,ids,max_cache_len=capacity);captures+=1
   else:decoder.prepare(ids);reuses+=1
   torch.cuda.synchronize();free_before,total=torch.cuda.mem_get_info();reserved_before=torch.cuda.memory_reserved()
   if a.trim_prefill_cache:torch.cuda.empty_cache()
   free_after,_=torch.cuda.mem_get_info();reserved_after=torch.cuda.memory_reserved()
   output=[int(decoder.token.item())];first=time.perf_counter()-start;decode_start=time.perf_counter()
   while len(output)<a.max_new_tokens and output[-1] not in eos:
    decoder.graph.replay();output.append(int(decoder.token.item()))
   torch.cuda.synchronize();decode_seconds=time.perf_counter()-decode_start;seconds=time.perf_counter()-start
   free,total=torch.cuda.mem_get_info()
   record={'request_index':index,'input_tokens':ids.shape[1],'output_tokens':len(output),'eos_reached':output[-1] in eos,'first_token_seconds':first,'request_seconds':seconds,'end_to_end_output_tokens_per_second':len(output)/seconds,'decode_after_first_tokens_per_second':(len(output)-1)/decode_seconds if len(output)>1 else None,'graph_capture_count':captures,'graph_reuse_count':reuses,'request_cache_len':capacity,'cache_capacity_changed':capacity_changed,'graph_released_for_long_prefill':release_for_long_prefill,'reserved_bytes_before_trim':reserved_before,'reserved_bytes_after_trim':reserved_after,'device_used_bytes_before_trim':total-free_before,'device_used_bytes_after_trim':total-free_after,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'device_used_bytes_at_observation':total-free,'device_total_bytes':total,'generated_token_ids':output,'text':tok.decode(output,skip_special_tokens=True)}
   result['requests'].append(record);save();print(json.dumps({k:v for k,v in record.items() if k!='generated_token_ids'},ensure_ascii=False),flush=True)
  assert captures+reuses==len(rows)
  if not a.auto_cache_len:assert captures==1 and reuses==len(rows)-1
  decoder.graph.reset();del decoder;gc.collect();torch.cuda.empty_cache()
 result['lazy_long_mask_stats']=dict(STATS);result['completed']=True;save()

if __name__=='__main__':main()
