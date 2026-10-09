"""Experimental fixed batch: shared output-head weight decode during reused prefills."""
import argparse,gc,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
from infer import manifest,ARTIFACTS


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--model',required=True,type=Path)
 p.add_argument('--requests',required=True,type=Path,help='UTF-8 JSONL, one prompt string per row.')
 p.add_argument('--output',required=True,type=Path)
 p.add_argument('--batch-size',choices=(2,4),type=int,default=4)
 p.add_argument('--max-new-tokens',type=int,default=64)
 a=p.parse_args();assert 1<=a.max_new_tokens<=64
 assert not a.output.exists(),'Choose a new output file; previous results are preserved.'
 rows=[json.loads(line) for line in a.requests.read_text(encoding='utf-8').splitlines() if line.strip()]
 assert rows and all(isinstance(r,dict) and isinstance(r.get('prompt'),str) and r['prompt'].strip() for r in rows)
 os.environ['BONSAI_BATCHED_PREFILL']='1'
 import torch
 from direct122_original_recovery_runtime_v38_expert_stream import assemble_controlled
 from split_decode122_native_math_v2 import bind
 from reusable122_shared_head_batch_decoder_v5_no_copy_checks import ReusableSharedHeadBatchDecoder as ReusableBatchDecoder
 from reusable122_graph_decoder_v1 import ReusableGraphDecoder
 assert torch.cuda.device_count()==1 and '5090' in torch.cuda.get_device_name(0)
 torch.set_num_threads(1);loaded=time.perf_counter();directory,report=manifest(a.model,'native4095')
 tok,model,_=assemble_controlled(directory,report);bind(model)
 tensors=[v for n,v in [*model.named_parameters(),*model.named_buffers()] if n.startswith(('model.language_model.','lm_head.'))]
 assert tensors and all(v.is_cuda for v in tensors)
 stores={v.untyped_storage().data_ptr():v.untyped_storage().nbytes() for v in tensors}
 assert sum(stores.values())==30171940608
 eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);assert eos and None not in eos
 result={'model':'Qwen3.5-122B-A10B','artifact_manifest_sha256':ARTIFACTS['native4095'][0],
  'model_load_seconds':time.perf_counter()-loaded,'unique_CUDA_text_storage_bytes':sum(stores.values()),
  'CPU_weight_offload':False,'batch_size_limit':a.batch_size,'max_cache_len':1536,
  'max_new_tokens':a.max_new_tokens,'eos_respected':True,'requests':[],'groups':[],'completed':False,
  'limits':'Experimental single worker, fixed B2/B4 groups, independent B1 prefills, cache1536, cap<=64. Greedy only. Finished rows stop emitting but compute until the group ends. No refill, HTTP, vLLM or continuous batching. Remainder3 is B2+B1, remainder1 uses the original single decoder. Experimental shared head on reused group prefills, body B1 shapes preserved. One graph resident at a time; same-size groups reuse it. Group timing excludes model load/tokenization/JSON writing, includes prefill/capture/reuse and CPU token collection. Original artifact quality scores remain below BF16.'}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 def save():
  f=a.output.with_suffix(a.output.suffix+'.tmp');f.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');f.replace(a.output)
 decoder=None;decoder_size=None;captures=0;reuses=0;offset=0
 with torch.inference_mode():
  while offset<len(rows):
   remaining=min(a.batch_size,len(rows)-offset);size=4 if remaining==4 else 2 if remaining>=2 else 1
   inputs=[]
   for row in rows[offset:offset+size]:
    text=tok.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=torch.tensor(tok.encode(text,add_special_tokens=False),device='cuda')[None]
    assert ids.shape[1]+a.max_new_tokens+4<1536,('Request exceeds experimental cache1536',offset+len(inputs))
    inputs.append(ids)
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
   if decoder is not None and size!=decoder_size:
    decoder.graph.reset();del decoder;decoder=None;gc.collect();torch.cuda.empty_cache()
   reused=decoder is not None
   if decoder is None:
    decoder=ReusableGraphDecoder(model,inputs[0],max_cache_len=1536) if size==1 else ReusableBatchDecoder(model,inputs,max_cache_len=1536)
    captures+=1;decoder_size=size
   else:
    decoder.prepare(inputs[0] if size==1 else inputs);reuses+=1
   torch.cuda.synchronize();outputs=[[int(v)] for v in decoder.token.cpu().flatten().tolist()]
   first=time.perf_counter()-begin;decode_begin=time.perf_counter();done=[v[-1] in eos or len(v)>=a.max_new_tokens for v in outputs];steps=0
   while not all(done):
    decoder.graph.replay();values=decoder.token.cpu().flatten().tolist();steps+=1
    for i,value in enumerate(values):
     if not done[i]:outputs[i].append(int(value));done[i]=value in eos or len(outputs[i])>=a.max_new_tokens
   torch.cuda.synchronize();decode_seconds=time.perf_counter()-decode_begin;seconds=time.perf_counter()-begin
   emitted=sum(map(len,outputs));free,total=torch.cuda.mem_get_info()
   group={'group_index':len(result['groups']),'first_request_index':offset,'batch_size':size,
    'input_tokens':[v.shape[1] for v in inputs],'output_tokens':emitted,'graph_reused':reused,
    'graph_capture_count':captures,'graph_reuse_count':reuses,'shared_head_prepare_count':getattr(decoder,'shared_head_prepare_count',0),'first_token_seconds':first,
    'decode_seconds':decode_seconds,'request_seconds':seconds,'decode_graph_steps':steps,
    'aggregate_useful_decode_tokens_per_second':(emitted-size)/decode_seconds if emitted>size else None,
    'aggregate_end_to_end_output_tokens_per_second':emitted/seconds,
    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
    'device_used_bytes_at_observation':total-free,'device_total_bytes':total}
   result['groups'].append(group)
   for i,output in enumerate(outputs):
    result['requests'].append({'request_index':offset+i,'group_index':group['group_index'],
     'input_tokens':inputs[i].shape[1],'output_tokens':len(output),'eos_reached':output[-1] in eos,
     'generated_token_ids':output,'text':tok.decode(output,skip_special_tokens=True)})
   save();print(json.dumps(group),flush=True);offset+=size
  decoder.graph.reset();del decoder;gc.collect();torch.cuda.empty_cache()
 result.update(completed=True,total_request_seconds=sum(g['request_seconds'] for g in result['groups']))
 result['aggregate_end_to_end_output_tokens_per_second']=sum(r['output_tokens'] for r in result['requests'])/result['total_request_seconds'];save()


if __name__=='__main__':main()
