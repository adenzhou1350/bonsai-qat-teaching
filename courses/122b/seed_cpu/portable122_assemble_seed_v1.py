"""CPU assembly of independently prepared122B ternary/dense4/embedding components.

Use --retained-only to extract the original361 non-quantized text tensors.
This produces an untrained expert-recovery seed, never a capability acceptance.
"""
import argparse,gc,hashlib,json,os,shutil
from pathlib import Path
import torch
from safetensors import safe_open

FORMAT='portable-direct-ternary122-dense4-rank8-seed-v1'
PREFIX='model.language_model.'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()
def tensor_sha(t):
    t=t.detach().contiguous().view(torch.uint8).reshape(-1);h=hashlib.sha256()
    for i in range(0,len(t),8*1024*1024):h.update(t[i:i+8*1024*1024].numpy().tobytes())
    return h.hexdigest()
def obj(path):return json.loads(Path(path).read_text())
def tensor_bytes(value):
    if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value,(tuple,list)):return sum(tensor_bytes(v) for v in value)
    return 0
def layout(source):
    cfg=obj(source/'config.json');cfg=cfg.get('text_config',cfg)
    assert (cfg['num_hidden_layers'],cfg['hidden_size'],cfg['num_experts'],cfg['moe_intermediate_size'],cfg['vocab_size'])==(48,3072,256,1024,248320)
    index=obj(source/'model.safetensors.index.json')['weight_map']
    text={n for n in index if n.startswith((PREFIX,'lm_head.'))};assert len(text)==831
    experts={f'{PREFIX}layers.{li}.mlp.experts.{attr}' for li in range(48) for attr in ('gate_up_proj','down_proj')}
    dense={}
    for li in range(48):
        prefix=f'{PREFIX}layers.{li}.'
        shared=['mlp.shared_expert.'+n for n in ('gate_proj','up_proj','down_proj')]
        linear=[f'linear_attn.{n}' for n in ('out_proj','in_proj_qkv','in_proj_z','in_proj_b','in_proj_a')]
        attention=[f'self_attn.{n}_proj' for n in ('q','k','v','o')]
        choice=linear if prefix+linear[0]+'.weight' in text else attention
        for name in shared+choice:
            original=prefix+name+'.weight';assert original in text
            dense[original]=original.removeprefix(PREFIX).removesuffix('.weight')
    dense['lm_head.weight']='lm_head'
    excluded=experts|set(dense)|{PREFIX+'embed_tokens.weight'}
    assert len(experts)==96 and len(dense)==373 and len(excluded)==470 and excluded<=text
    retained=text-excluded;assert len(retained)==361
    return index,experts,dense,retained
def read_retained(source,index,names,expected=None):
    values={}
    for name in sorted(names):
        with safe_open(str(source/index[name]),framework='pt',device='cpu') as f:value=f.get_tensor(name)
        assert value.dtype in (torch.bfloat16,torch.float32) and torch.isfinite(value).all()
        if expected is not None:assert tensor_sha(value)==expected[name],('Original retained tensor differs from source calibration',name)
        values[name]=value
    assert len(values)==361 and tensor_bytes(values)==79971840
    return values
def component_path(root,name):
    assert Path(name).name==name and name not in ('','.','..')
    p=root/name;assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root.resolve())
    return p
def admit_components(source,index,experts,dense,eroot,droot,broot):
    e=obj(eroot/'summary.json');d=obj(droot/'summary.json');b=obj(broot/'summary.json');ish=sha(source/'model.safetensors.index.json')
    assert e['passed'] and e['format']=='portable-direct-lloyd8-ternary122-experts-v1' and e['expert_only'] and e['actual_optimizer_updates']==0 and len(e['matrices'])==96
    assert d['passed'] and d['CPU_only'] and d['dense_only'] and d['all48_and_head_calibrated'] and d['completed_layers']==48 and len(d['layers'])==48 and len(d['matrices'])==373,'Complete48-layer plus output-head dense calibration required'
    assert b['passed'] and b['cpu_only'] and b['actual_optimizer_updates']==128 and b['rank']==16 and b['stored_training_and_serialized_forward_bitwise_equal'] and b['packed_base_unchanged'] and b['source_original_parameters_unchanged'] and b['full_embedding_shape']==[248320,3072]
    assert e['source_index_sha256']==d['source_index_sha256']==b['source_index_sha256']==ish
    assert b['data']['tokens_sha256']==d['tokens_file_sha256'] and d['calibration_train_blocks']==512 and d['calibration_positions_per_block']==512 and d['validation_blocks']==16
    original={}
    for li,row in enumerate(d['layers']):
        assert row['layer']==li and row['source_parameters_unchanged']
        for name,h in row['source_parameters_sha256'].items():original[f'{PREFIX}layers.{li}.{name}']=h
    original[PREFIX+'embed_tokens.weight']=b['source_embedding_sha256']
    original[PREFIX+'norm.weight']=d['output_head']['original_norm_sha256']
    original['lm_head.weight']=d['output_head']['source_weight_sha256']
    assert set(original)=={n for n in index if n.startswith((PREFIX,'lm_head.'))} and len(original)==831
    enames=[]
    for row in e['matrices']:
        name=PREFIX+row['module']+'.'+row['attr'];enames.append(name)
        assert name in experts and row['packing_format']=='own-gsq-five-trits-expert-bank-v1' and row['source_tensor_sha256']==original[name]
        assert sha(component_path(eroot,row['file']))==row['sha256']
    assert len(set(enames))==96 and set(enames)==experts
    dnames=[]
    for row in d['matrices']:
        name='lm_head.weight' if row['module']=='lm_head' else PREFIX+row['module']+'.weight';dnames.append(name)
        assert name in dense and row['packing_format']=='own-affine-dense4-v1' and row['attr']=='weight'
        assert sha(component_path(droot,row['file']))==row['sha256']
    assert len(set(dnames))==373 and set(dnames)==set(dense)
    assert sha(component_path(broot,'embedding.pt'))==b['artifact_sha256']
    return e,d,b,original
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--experts',type=Path);p.add_argument('--dense',type=Path);p.add_argument('--embedding',type=Path);p.add_argument('--retained-only',action='store_true');a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    torch.set_num_threads(8);source=a.model.resolve();out=a.output.resolve();assert not out.exists()
    index,experts,dense,retained_names=layout(source);index_sha=sha(source/'model.safetensors.index.json')
    if a.retained_only:
        assert a.experts is a.dense is a.embedding is None
        values=read_retained(source,index,retained_names);out.mkdir(parents=True,exist_ok=False);torch.save(values,out/'retained.pt')
        report={'passed':True,'format':'portable122-original-retained361-CPU-v1','CPU_only':True,'source_index_sha256':index_sha,'retained_tensor_sha256':{n:tensor_sha(t) for n,t in values.items()},'retained_tensor_bytes':tensor_bytes(values),'retained_sha256':sha(out/'retained.pt'),'script_sha256':sha(Path(__file__)),'complete_seed_assembled':False,'quality_accepted':False}
    else:
        assert all(x is not None for x in (a.experts,a.dense,a.embedding))
        eroot,droot,broot=(x.resolve() for x in (a.experts,a.dense,a.embedding))
        e,d,b,original=admit_components(source,index,experts,dense,eroot,droot,broot)
        values=read_retained(source,index,retained_names,original)
        required=sum(component_path(r,row['file']).stat().st_size for r,rows in ((eroot,e['matrices']),(droot,d['matrices'])) for row in rows)+component_path(broot,'embedding.pt').stat().st_size+tensor_bytes(values)
        assert out.parent.is_dir() and shutil.disk_usage(out.parent).free>required+8*1024**3
        out.mkdir(exist_ok=False);torch.save(values,out/'retained.pt');matrices=[]
        for prefix,root,rows in (('expert',eroot,e['matrices']),('dense',droot,d['matrices'])):
            for i,row in enumerate(rows):
                path=component_path(root,row['file']);name=f'{prefix}-{i:03}.pt';assert not (out/name).exists()
                shutil.copyfile(path,out/name);assert sha(out/name)==sha(path)==row['sha256']
                bank=torch.load(out/name,map_location='cpu',weights_only=True)
                assert bank['shape']==row['shape'];entry={**row,'file':name,'sha256':sha(out/name),'tensor_bytes':tensor_bytes(bank)}
                if prefix=='expert':assert bank['format']=='own_gsq_five_trits_expert_bank_v1' and len(bank['states'])==256 and 'retained_factor' not in bank
                else:entry['name']=row['module']+'.weight'
                matrices.append(entry);del bank;gc.collect()
                print(json.dumps({'stage':'copy_verified_component','kind':prefix,'completed':i+1,'total':len(rows)}),flush=True)
        shutil.copyfile(broot/'embedding.pt',out/'embedding.pt');assert sha(out/'embedding.pt')==b['artifact_sha256']
        estate=torch.load(out/'embedding.pt',map_location='cpu',weights_only=True)
        embedding={'file':'embedding.pt','sha256':sha(out/'embedding.pt'),'tensor_bytes':tensor_bytes(estate),'initialization_optimizer_updates':128};del estate
        lineage={'format':'portable122-component-lineage-v1','expert_summary':e,'dense_summary':d,'embedding_summary':b,'component_manifest_sha256':{'experts':sha(eroot/'summary.json'),'dense':sha(droot/'summary.json'),'embedding':sha(broot/'summary.json')},'source_parameter_sha256':original,'expert_recovery_training_completed':False}
        (out/'lineage.json').write_text(json.dumps(lineage,indent=2))
        report={'passed':True,'format':FORMAT,'cpu_only':True,'source_model':str(source),'source_index_sha256':index_sha,'layers':list(range(48)),'original_model_layers':48,'full_model_exported':True,'joint_component_training':False,'actual_optimizer_updates':0,'expert_seed_gradient_updates':0,'embedding_initialization_optimizer_updates':128,'matrices':matrices,'embedding':embedding,'retained_sha256':sha(out/'retained.pt'),'retained_tensor_sha256':{n:tensor_sha(t) for n,t in values.items()},'retained_tensor_bytes':tensor_bytes(values),'lineage_file':'lineage.json','lineage_sha256':sha(out/'lineage.json'),'actual_seed_static_tensor_bytes':sum(r['tensor_bytes'] for r in matrices)+embedding['tensor_bytes']+tensor_bytes(values),'script_sha256':sha(Path(__file__)),'quality_accepted':False,'limits':'Complete portable CPU initialization only: frozen source-Lloyd8 ternary experts, original-activation GPTQ dense4, RTN4 embedding with128 TRAIN-only rank16 correction updates, original retained361 tensors. Expert rank8 recovery, independent capabilities and single5090 inference of this generated artifact remain required.'}
    restored=torch.load(out/'retained.pt',map_location='cpu',weights_only=True)
    assert set(restored)==set(values) and all(torch.equal(restored[n],t) and restored[n].dtype==t.dtype for n,t in values.items())
    assert sha(source/'model.safetensors.index.json')==index_sha and not torch.cuda.is_initialized()
    for n in ('config.json','tokenizer.json','tokenizer_config.json','generation_config.json'):
        path=source/n;assert path.is_file();shutil.copyfile(path,out/n);assert sha(path)==sha(out/n)
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({'passed':True,'retained_tensor_count':361,'complete_seed_assembled':not a.retained_only,'expert_recovery_training_completed':False,'quality_accepted':False}))
if __name__=='__main__':main()
