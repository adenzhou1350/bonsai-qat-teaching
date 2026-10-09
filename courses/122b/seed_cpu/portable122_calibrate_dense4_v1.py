"""Original-source CPU GPTQ4, all48 layers and output head; expert-only seed separate.

Input JSON: {"train":512 lists of513 token IDs,"validation":16 lists of513 IDs}.
Only train rows update Hessians. Partial --layers runs explicitly omit the head.
This exports dense weights only, not a complete deployable or trained model.
"""
import argparse,gc,hashlib,json,os,shutil,time
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as q
from portable122_dense_capture_v1 import Dense4Calibration

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()

def tensor_sha(t):
    t=t.detach().contiguous().view(torch.uint8).reshape(-1);h=hashlib.sha256()
    for i in range(0,len(t),16*1024*1024):h.update(t[i:i+16*1024*1024].numpy().tobytes())
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--tokens',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--layers',type=int,default=48)
    p.add_argument('--threads',type=int,default=8);a=p.parse_args()
    assert 1<=a.layers<=48 and 1<=a.threads<=32
    torch.set_num_threads(a.threads);assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'Run with CUDA_VISIBLE_DEVICES empty.'
    cfg=AutoConfig.from_pretrained(a.model,local_files_only=True).text_config
    assert cfg.num_hidden_layers==48 and cfg.hidden_size==3072 and cfg.num_experts==256 and cfg.moe_intermediate_size==1024
    cfg._attn_implementation='sdpa';cfg._experts_implementation='grouped_mm'
    q.FusedRMSNormGated=None;q.causal_conv1d_fn=None;q.chunk_gated_delta_rule=None;q.fused_recurrent_gated_delta_rule=None
    tokens=json.loads(a.tokens.read_text());assert len(tokens['train'])==512 and len(tokens['validation'])==16
    rows=tokens['train']+tokens['validation']
    assert all(len(ids)==513 and all(type(v) is int and 0<=v<cfg.vocab_size for v in ids) for ids in rows)
    assert len({tuple(x) for x in rows})==528,'Duplicate calibration blocks'
    index_file=a.model/'model.safetensors.index.json';index=json.loads(index_file.read_text())['weight_map']
    available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    assert available>24*1024**3,'CPU calibration requires at least24GiB available RAM'
    assert a.output.parent.is_dir() and shutil.disk_usage(a.output.parent).free>8*1024**3
    a.output.mkdir(exist_ok=False);start=time.monotonic()
    files=('portable122_calibrate_dense4_v1.py','portable122_dense_capture_v1.py','portable122_gptq4_v1.py','affine4_cpu.py')
    source_dir=Path(__file__).resolve().parent;policy={n:sha(source_dir/n) for n in files}
    report={'passed':False,'CPU_only':True,'dense_only':True,'source_index_sha256':sha(index_file),'tokens_file_sha256':sha(a.tokens),'source_sha256':policy,'calibration_train_blocks':512,'calibration_positions_per_block':512,'validation_blocks':16,'layers':[],'matrices':[],'full_model_exported':False,'quality_accepted':False}
    def progress(stage,**kw):
        value={'pid':os.getpid(),'stage':stage,'seconds':time.monotonic()-start,**kw}
        f=a.output/'progress.json';tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(f)
        print(json.dumps(value),flush=True)
    def tensor(name):
        with safe_open(str(a.model/index[name]),framework='pt',device='cpu') as f:return f.get_tensor(name)
    def write_matrices(states,modules):
        for name,state in states.items():
            file=f'dense-{len(report["matrices"]):03}.pt';torch.save(state,a.output/file)
            report['matrices'].append({'module':modules[name],'attr':'weight','shape':state['shape'],'file':file,'sha256':sha(a.output/file),'packing_format':'own-affine-dense4-v1'})
    progress('loading_original_embedding')
    embedding=tensor('model.language_model.embed_tokens.weight');embedding_hash=tensor_sha(embedding)
    with torch.inference_mode():original=[torch.nn.functional.embedding(torch.tensor(ids[:-1])[None],embedding) for ids in rows]
    assert tensor_sha(embedding)==embedding_hash;del embedding;gc.collect()
    rope=q.Qwen3_5MoeTextRotaryEmbedding(cfg,device=torch.device('cpu'))
    for li in range(a.layers):
        progress('loading_original_layer',layer=li);prefix=f'model.language_model.layers.{li}.'
        values={n[len(prefix):]:tensor(n) for n in index if n.startswith(prefix)}
        with torch.device('meta'):layer=q.Qwen3_5MoeDecoderLayer(cfg,li)
        layer.load_state_dict(values,strict=True,assign=True);layer.requires_grad_(False);layer.eval()
        before={n:tensor_sha(v) for n,v in layer.named_parameters()};capture=Dense4Calibration(layer)
        outputs=[]
        with torch.inference_mode():
            for bi,x in enumerate(original):
                capture.enabled=bi<512;pos=torch.arange(x.shape[1])[None]
                with torch.autocast('cpu',dtype=torch.bfloat16):outputs.append(layer(x,position_embeddings=rope(x,pos),position_ids=pos,past_key_values=None))
                if (bi+1)%16==0:progress('original_source_dense_capture',layer=li,completed_blocks=bi+1,train_blocks=min(bi+1,512))
        states,metrics=capture.fit(lambda row:progress('original_dense_GPTQ_fit',layer=li,last_matrix=row))
        write_matrices(states,{name:f'layers.{li}.{name}' for name in states})
        assert all(tensor_sha(v)==before[n] for n,v in layer.named_parameters())
        validation={'token_ids':tokens['validation'],'source_output_states':outputs[512:]};vp=a.output/f'layer-{li:02}-validation.pt';torch.save(validation,vp)
        report['layers'].append({'layer':li,'source_parameters_sha256':before,'source_parameters_unchanged':True,'dense_calibration':metrics,'validation_file':vp.name,'validation_file_sha256':sha(vp)})
        original=outputs;progress('layer_completed',completed_layers=li+1)
        del layer,values,states,capture,outputs,validation;gc.collect()
    if a.layers==48:
        progress('original_final_head_loading')
        norm_weight=tensor('model.language_model.norm.weight');norm_hash=tensor_sha(norm_weight)
        norm=q.Qwen3_5MoeRMSNorm(cfg.hidden_size,eps=cfg.rms_norm_eps);norm.load_state_dict({'weight':norm_weight},assign=True);norm.requires_grad_(False);norm.eval()
        head_weight=tensor('lm_head.weight');head_hash=tensor_sha(head_weight)
        with torch.device('meta'):holder=torch.nn.Module();holder.head=torch.nn.Linear(cfg.hidden_size,len(head_weight),bias=False)
        holder.head.load_state_dict({'weight':head_weight},assign=True);holder.requires_grad_(False)
        capture=Dense4Calibration(holder);assert set(capture.chosen)=={'head'};capture.enabled=True
        with torch.inference_mode():
            for bi,x in enumerate(original[:512]):
                normalized=norm(x);capture.hook('head')(holder.head,(normalized,),None)
                if (bi+1)%32==0:progress('original_final_head_capture',completed_train_blocks=bi+1)
        states,metrics=capture.fit(lambda row:progress('original_final_head_GPTQ_fit',last_matrix=row))
        write_matrices(states,{'head':'lm_head'})
        assert tensor_sha(norm_weight)==norm_hash and tensor_sha(head_weight)==head_hash
        report['output_head']={'calibration':metrics,'source_weight_sha256':head_hash,'original_norm_sha256':norm_hash}
    assert all(sha(source_dir/n)==h for n,h in policy.items()) and not torch.cuda.is_initialized()
    report.update(passed=True,completed_layers=a.layers,all48_and_head_calibrated=a.layers==48,seconds=time.monotonic()-start,limits='Original-source dense-only calibration; no expert or embedding initialization and no rank8 recovery. Partial layer runs omit output-head calibration. No deployment or capability claim. Only512 TRAIN blocks update Hessians;16 validation outputs are read-only controls.')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2));progress('completed',completed_layers=a.layers,all48_and_head_calibrated=a.layers==48)

if __name__=='__main__':main()
