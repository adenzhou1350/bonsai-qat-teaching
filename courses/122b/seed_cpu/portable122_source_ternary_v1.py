"""CPU expert PTQ from original Qwen3.5-122B tensors; no experiment directories.

This prepares expert banks only. Dense4 calibration, embedding correction,
teacher states and rank8 recovery are separate stages, not supplied here.
"""
import argparse
import gc
import hashlib
import json
import math
import shutil
from pathlib import Path
import torch
from safetensors import safe_open

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()

def tensor_sha(value):
    flat=value.detach().contiguous().view(torch.uint8).reshape(-1)
    h=hashlib.sha256()
    for i in range(0,len(flat),8*1024*1024):h.update(flat[i:i+8*1024*1024].numpy().tobytes())
    return h.hexdigest()

def fit_expert(weight):
    """Same BF16 Lloyd8/group128 grid as the completed native4095 experiment."""
    assert weight.device.type=='cpu' and weight.dtype==torch.bfloat16 and weight.ndim==2
    n,k=weight.shape
    assert k%128==0 and torch.isfinite(weight).all()
    groups=weight.float().reshape(n,k//128,128)
    scale=(groups.abs().mean(-1)*1.5).bfloat16()
    for _ in range(8):
        codes=(groups/scale.float().clamp_min(1e-30)[...,None]).round().clamp(-1,1)
        scale=((groups*codes).sum(-1)/codes.square().sum(-1).clamp_min(1)).bfloat16()
    codes=(groups/scale.float().clamp_min(1e-30)[...,None]).round().clamp(-1,1).reshape(n,k).to(torch.int16)
    digits=codes+1
    padded=math.ceil(k/5)*5
    if padded!=k:digits=torch.nn.functional.pad(digits,(0,padded-k),value=1)
    powers=torch.tensor([1,3,9,27,81],dtype=torch.int16)
    packed=(digits.reshape(n,-1,5)*powers).sum(-1).to(torch.uint8)
    restored=((packed.to(torch.int16)[...,None]//powers)%3-1).reshape(n,-1)[:,:k]
    assert torch.equal(restored,codes) and torch.isfinite(scale).all()
    hard=(codes.bfloat16()*scale.repeat_interleave(128,dim=-1)).bfloat16()
    error=float((hard.float()-weight.float()).square().sum())
    denominator=float(weight.float().square().sum())
    state={'format':'own_gsq_five_trits_per_byte_v1','shape':[n,k],'codes':packed,'scales':scale,
           'exact_source_bf16_grid':True,'grid_origin':'own_direct_lloyd8_source_ptq_no_gsq_gradient_updates'}
    return state,error,denominator

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=8)
    args=parser.parse_args()
    assert args.threads>0 and not torch.cuda.is_initialized()
    torch.set_num_threads(args.threads)
    source=args.model.resolve(); output=args.output.resolve()
    index_file=source/'model.safetensors.index.json'
    index=json.loads(index_file.read_text())['weight_map']
    config=json.loads((source/'config.json').read_text())
    text=config.get('text_config',config)
    assert text['num_hidden_layers']==48 and text['hidden_size']==3072 and text['num_experts']==256
    assert text['moe_intermediate_size']==1024
    output.mkdir(parents=True,exist_ok=False)
    summary={'format':'portable-direct-lloyd8-ternary122-experts-v1','passed':False,
             'source_index_sha256':file_sha(index_file),'quantizer_sha256':file_sha(Path(__file__)),
             'matrices':[],'actual_optimizer_updates':0,'expert_only':True,
             'limits':'CPU PTQ expert initialization only; no dense4, embedding, teacher, recovery training, complete deployable model or capability acceptance.'}
    for layer in range(48):
        for attr,shape in (('gate_up_proj',[256,2048,3072]),('down_proj',[256,3072,1024])):
            name=f'model.language_model.layers.{layer}.mlp.experts.{attr}'
            states=[]; original_hash=hashlib.sha256(); error=denominator=0.
            with safe_open(str(source/index[name]),framework='pt',device='cpu') as handle:
                view=handle.get_slice(name)
                assert view.get_shape()==shape
                for expert in range(256):
                    value=view[expert].contiguous()
                    original_hash.update(value.view(torch.uint8).numpy().tobytes())
                    state,e,d=fit_expert(value)
                    states.append(state); error+=e; denominator+=d
            bank={'format':'own_gsq_five_trits_expert_bank_v1','shape':shape,'states':states,
                  'origin':'direct_lloyd8_source_ptq_not_official_gsq'}
            file=output/f'layer-{layer:02}-{attr}.pt'
            torch.save(bank,file)
            summary['matrices'].append({'module':f'layers.{layer}.mlp.experts','attr':attr,'file':file.name,
                'packing_format':'own-gsq-five-trits-expert-bank-v1','shape':shape,'sha256':file_sha(file),
                'source_tensor_sha256':original_hash.hexdigest(),
                'source_weight_relative_mse':error/max(denominator,1e-30),
                'tensor_bytes':sum(v.numel()*v.element_size() for s in states for v in s.values() if isinstance(v,torch.Tensor))})
            del states,bank;gc.collect()
            (output/'progress.json').write_text(json.dumps({'completed_banks':len(summary['matrices']),'planned_banks':96},indent=2))
            print(f'Layer {layer+1}/48 {attr}',flush=True)
    assert len(summary['matrices'])==96 and not torch.cuda.is_initialized()
    assert file_sha(index_file)==summary['source_index_sha256']
    for name in ('config.json','tokenizer.json','tokenizer_config.json','generation_config.json'):
        if (source/name).exists():shutil.copyfile(source/name,output/name)
    summary['passed']=True
    (output/'summary.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__':main()
