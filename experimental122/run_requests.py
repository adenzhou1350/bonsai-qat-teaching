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
 tensors=[v for n,v in [*model.named_parameters(),*model.named_buffers()] if n.startswith(('model.language_model.','lm_head.'))]
 assert tensors and all(v.is_cuda for v in tensors)
 stores={v.untyped_storage().data_ptr():v.untyped_storage().nbytes() for v in tensors}
 eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);assert eos and None not in eos
 result={'model':'Qwen3.5-122B-A10B','artifact_manifest_sha256':ARTIFACTS[a.artifact_version][0],'model_load_seconds':time.perf_counter()-loaded,'unique_CUDA_text_storage_bytes':sum(stores.values()),'CPU_weight_offload':False,'batch_size':1,'max_cache_len':a.max_cache_len,'max_new_tokens':a.max_new_tokens,'eos_respected':True,'requests':[],'completed':False,'limits':'Resident single worker, serial requests, greedy generation; no concurrent batching, streaming API or vLLM. Request timing excludes model load, includes prefill and first graph capture. Graph reuse was verified on short prompts at max_cache_len1536; other capacities require separate validation. Original artifact quality scores remain below BF16.'}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 def save():
  f=a.output.with_suffix(a.output.suffix+'.tmp');f.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');f.replace(a.output)
 decoder=None
 with torch.inference_mode():
  for index,row in enumerate(rows):
   text=tok.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   ids=torch.tensor(tok.encode(text,add_special_tokens=False),device='cuda')[None]
   assert ids.shape[1]+a.max_new_tokens+4<a.max_cache_len,('Request exceeds fixed cache capacity',index)
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
   if decoder is None:decoder=ReusableGraphDecoder(model,ids,max_cache_len=a.max_cache_len)
   else:decoder.prepare(ids)
   output=[int(decoder.token.item())];first=time.perf_counter()-start;decode_start=time.perf_counter()
   while len(output)<a.max_new_tokens and output[-1] not in eos:
    decoder.graph.replay();output.append(int(decoder.token.item()))
   torch.cuda.synchronize();decode_seconds=time.perf_counter()-decode_start;seconds=time.perf_counter()-start
   free,total=torch.cuda.mem_get_info()
   record={'request_index':index,'input_tokens':ids.shape[1],'output_tokens':len(output),'eos_reached':output[-1] in eos,'first_token_seconds':first,'request_seconds':seconds,'end_to_end_output_tokens_per_second':len(output)/seconds,'decode_after_first_tokens_per_second':(len(output)-1)/decode_seconds if len(output)>1 else None,'graph_capture_count':decoder.capture_count,'graph_reuse_count':decoder.reuse_count,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'device_used_bytes_at_observation':total-free,'device_total_bytes':total,'generated_token_ids':output,'text':tok.decode(output,skip_special_tokens=True)}
   result['requests'].append(record);save();print(json.dumps({k:v for k,v in record.items() if k!='generated_token_ids'},ensure_ascii=False),flush=True)
  assert decoder.capture_count==1 and decoder.reuse_count==len(rows)-1
  decoder.graph.reset();del decoder;gc.collect();torch.cuda.empty_cache()
 result['completed']=True;save()

if __name__=='__main__':main()
