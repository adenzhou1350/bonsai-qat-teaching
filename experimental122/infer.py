"""Single-request experimental 122B inference; requires existing packed weights."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
EXPECTED_MANIFEST='5c9683297d71c1dbdd1c32a8a749aa1911c857fb9ef1a788dfe794230abd08a9'
ARTIFACTS={
    'old':(EXPECTED_MANIFEST,'research-direct-ternary122-dense4-original-teacher-rank8-recovery-v1'),
    'native4095':('dc3a9ba1a8bd9bcdd229b3c259cc5d676719861946a5ed2d90b39ac2e196def1','research-direct-ternary122-dense4-native4095-teacher-rank8-recovery-v2'),
}

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(8*1024*1024):digest.update(block)
    return digest.hexdigest()

def manifest(directory,artifact_version='old'):
    directory=Path(directory).resolve();path=directory/'summary.json'
    expected,format_tag=ARTIFACTS[artifact_version]
    if sha(path)!=expected:raise ValueError(f'This entry requires the pinned {artifact_version} 4096-step experimental artifact.')
    report=json.loads(path.read_text())
    assert report['format']==format_tag and report['passed'] and report['full_model_exported'] and report['layers']==list(range(48)) and report['actual_optimizer_updates']==4096
    if artifact_version=='native4095':assert report['teacher_input_positions']==4095 and report['serialized_replay_bitwise_equal'] and report['frozen_components_unchanged']
    assert sha(directory/'retained.pt')==report['retained_sha256']
    for row in [*report['matrices'],report['embedding']]:
        path=(directory/row['file']).resolve();assert path.is_relative_to(directory) and sha(path)==row['sha256']
    return directory,report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True,type=Path);p.add_argument('--artifact-version',choices=ARTIFACTS,default='old');p.add_argument('--prompt',default='用中文解释什么是模型量化。');p.add_argument('--max-new-tokens',type=int,default=128);p.add_argument('--benchmark',action='store_true');p.add_argument('--contexts',type=int,nargs='+',default=[128,512,4096,8192,16384]);p.add_argument('--output',type=Path);a=p.parse_args()
    assert 1<=a.max_new_tokens<=1024 and all(1<=v<=16384 for v in a.contexts)
    os.environ['BONSAI_BATCHED_PREFILL']='1'
    import torch
    from direct122_original_recovery_runtime_v38_expert_stream import assemble_controlled
    from gsq_graph_decoder_v3_cache_snapshot import GraphDecoder
    assert torch.cuda.device_count()==1 and '5090' in torch.cuda.get_device_name(0),'Select one RTX 5090 with CUDA_VISIBLE_DEVICES.'
    torch.set_num_threads(1);directory,report=manifest(a.model,a.artifact_version);tok,model,_=assemble_controlled(directory,report)
    text_tensors=[v for n,v in [*model.named_parameters(),*model.named_buffers()] if n.startswith(('model.language_model.','lm_head.'))];assert text_tensors and all(v.is_cuda for v in text_tensors)
    stores={v.untyped_storage().data_ptr():v.untyped_storage().nbytes() for v in text_tensors}
    result={'model':'Qwen3.5-122B-A10B','artifact_version':a.artifact_version,'artifact_manifest_sha256':ARTIFACTS[a.artifact_version][0],'artifact_optimizer_updates':4096,'engineering_only':True,'quality_accepted':False,'cpu_weight_offload':False,'all_text_tensors_on_CUDA':True,'unique_CUDA_text_storage_bytes':sum(stores.values()),'benchmarks':[],'limits':'Quality-rejected experimental weights. Single request, text only. Fixed64 benchmark excludes prefill and capture and continues past EOS. Not a standard Transformers/vLLM checkpoint.'}
    with torch.inference_mode():
        if a.benchmark:
            import gc
            pattern=tok.encode('This is a controlled timing context for a language model. ',add_special_tokens=False)
            for context in a.contexts:
                ids=torch.tensor((pattern*((context+len(pattern)-1)//len(pattern)))[:context],device='cuda')[None];torch.cuda.reset_peak_memory_stats();decoder=GraphDecoder(model,ids,max_cache_len=context+128)
                torch.cuda.synchronize();start=time.perf_counter();decoder.prepare(ids);torch.cuda.synchronize();prefill=time.perf_counter()-start
                graph=[];torch.cuda.synchronize();start=time.perf_counter()
                for _ in range(64):decoder.graph.replay();graph.append(decoder.token.clone())
                torch.cuda.synchronize();seconds=time.perf_counter()-start;graph_ids=torch.cat(graph,-1);decoder.prepare(ids);eager=[]
                for _ in range(64):decoder.step();eager.append(decoder.token.clone())
                torch.cuda.synchronize();assert torch.equal(graph_ids,torch.cat(eager,-1))
                row={'context_tokens':context,'fixed_new_tokens':64,'decode_tokens_per_second':64/seconds,'prepare_seconds_includes_prefill_and_capture':prefill,'graph_static_token_ids_equal':True,'peak_allocated_bytes':torch.cuda.max_memory_allocated()};result['benchmarks'].append(row);print(json.dumps(row),flush=True)
                del decoder,ids,graph_ids,graph,eager;gc.collect();torch.cuda.empty_cache()
        else:
            prompt=tok.apply_chat_template([{'role':'user','content':a.prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False);ids=torch.tensor(tok.encode(prompt,add_special_tokens=False),device='cuda')[None];assert ids.shape[1]+a.max_new_tokens<=16384
            eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);torch.cuda.reset_peak_memory_stats();decoder=GraphDecoder(model,ids,max_cache_len=ids.shape[1]+a.max_new_tokens+128)
            torch.cuda.synchronize();start=time.perf_counter();generated=decoder.generate_ids(ids,a.max_new_tokens,eos);torch.cuda.synchronize();result.update(generated_token_ids=generated,text=tok.decode(generated,skip_special_tokens=True),request_seconds_includes_prepare=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated());print(result['text'],flush=True)
    result['completed']=True
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('generated_token_ids','text')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
