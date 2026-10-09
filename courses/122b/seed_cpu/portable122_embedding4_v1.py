"""Standalone122B full-vocabulary affine4 RTN and128 TRAIN-only rank16 updates.

Use CUDA_VISIBLE_DEVICES empty. Input train512/validation16 arrays of513 IDs.
Embedding component only; not a complete trained or deployable model.
"""
import argparse,hashlib,json,math,os,time
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors import safe_open
from affine4_cpu import pack,unpack

def tensor_sha(value):
    flat=value.detach().contiguous().view(torch.uint8).reshape(-1);digest=hashlib.sha256()
    for i in range(0,len(flat),16*1024*1024):digest.update(flat[i:i+16*1024*1024].numpy().tobytes())
    return digest.hexdigest()

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def lookup(state,ids):
    flat=ids.reshape(-1);subset={**state,'shape':[len(flat),state['shape'][1]],'codes':state['codes'][flat],'scales':state['scales'][flat],'zeros':state['zeros'][flat]};return unpack(subset).reshape(*ids.shape,state['shape'][1])

def forward(state,ids,A,B):return lookup(state,ids)+F.linear(A[ids].to(torch.bfloat16),B.to(torch.bfloat16))*.125

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--tokens',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    source=a.model;out=a.output;out.mkdir(exist_ok=False);began=time.perf_counter();torch.set_num_threads(8);assert not torch.cuda.is_initialized()
    index_path=source/'model.safetensors.index.json';index=json.loads(index_path.read_text())['weight_map'];name='model.language_model.embed_tokens.weight'
    with safe_open(str(source/index[name]),framework='pt',device='cpu') as file:weight=file.get_tensor(name)
    assert weight.dtype==torch.bfloat16 and list(weight.shape)==[248320,3072]
    original_sha=tensor_sha(weight);script_dir=Path(__file__).resolve().parent
    policy={n:sha(script_dir/n) for n in ('portable122_embedding4_v1.py','affine4_cpu.py')}
    tokens=json.loads(a.tokens.read_text());assert len(tokens['train'])==512 and len(tokens['validation'])==16
    rows=tokens['train']+tokens['validation'];assert all(len(x)==513 and all(type(v) is int and 0<=v<len(weight) for v in x) for x in rows)
    assert len({tuple(x) for x in rows})==528
    data={'tokens_sha256':sha(a.tokens),'train_blocks':512,'validation_blocks':16,'input_positions_per_block':512}
    train_ids=torch.tensor([t for ids in rows[:512] for t in ids[:-1]]);val_ids=[torch.tensor(ids[:-1])[None] for ids in rows[512:]];vocab,k=weight.shape
    assert len(train_ids)==512*512 and int(train_ids.min())>=0 and int(train_ids.max())<vocab;parts=[];squared_error=0.;denominator=0.
    def save(stage,**event):
        record={'stage':stage,'seconds':time.perf_counter()-began,**event};(out/'progress.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
    with torch.inference_mode():
        for start in range(0,vocab,4096):
            w=weight[start:start+4096];x=w.float().reshape(len(w),k//128,128);lo=torch.minimum(x.amin(-1),torch.zeros(len(w),k//128));hi=torch.maximum(x.amax(-1),torch.zeros(len(w),k//128));scale=(hi-lo).clamp_min(1e-5)/15;zero=(-lo/scale).round();codes=((x/scale[...,None]).round()+zero[...,None]).clamp(0,15);state=pack(codes,scale.to(torch.bfloat16),zero.to(torch.bfloat16),list(w.shape));hard=unpack(state);squared_error+=float((hard.float()-w.float()).square().sum());denominator+=float(w.float().square().sum());parts.append(state);save('full_vocabulary_rtn_packing',packed_rows=start+len(w),total_rows=vocab)
    # Leave inference tensors behind before constructing gradient inputs.
    state={**parts[0],'shape':[vocab,k],'role':'embedding','codes':torch.cat([p['codes'] for p in parts]).clone(),'scales':torch.cat([p['scales'] for p in parts]).clone(),'zeros':torch.cat([p['zeros'] for p in parts]).clone()};del parts;baseline_hash={n:tensor_sha(state[n]) for n in ('codes','scales','zeros')};generator=torch.Generator().manual_seed(773);rank=16;A=torch.nn.Parameter(torch.zeros(vocab,rank));B=torch.nn.Parameter(torch.randn(k,rank,generator=generator)/math.sqrt(rank));optimizer=torch.optim.Adam([A,B],lr=.001);events=[]
    def errors(a_factors,b_factors):
        results=[]
        with torch.inference_mode():
            for ids in val_ids:
                target=F.embedding(ids,weight);base=lookup(state,ids);prediction=forward(state,ids,a_factors,b_factors);denom=target.float().square().mean().clamp_min(1e-12);results.append({'base_relative_mse':float((base.float()-target.float()).square().mean()/denom),'corrected_relative_mse':float((prediction.float()-target.float()).square().mean()/denom)})
        return results
    before=errors(A,B)
    for step in range(128):
        positions=torch.randint(len(train_ids),(256,),generator=generator);ids=train_ids[positions];target=F.embedding(ids,weight).detach();prediction=forward(state,ids,A,B);loss=(prediction.float()-target.float()).square().mean()/target.float().square().mean().clamp_min(1e-12);assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();assert A.grad is not None and B.grad is not None and torch.isfinite(A.grad).all() and torch.isfinite(B.grad).all();norm=torch.nn.utils.clip_grad_norm_([A,B],1.);assert torch.isfinite(norm);optimizer.step()
        if step in (0,31,63,95,127):event={'step':step+1,'train_loss':float(loss.detach()),'finite_connected_gradients':True,'gradient_norm_before_clip':float(norm)};events.append(event);save('train_only_embedding_correction',**event)
    state.update(embedding_A=A.detach().to(torch.bfloat16),embedding_B=B.detach().to(torch.bfloat16),lora_scale=.125);seen=torch.zeros(vocab,dtype=torch.bool);seen[train_ids]=True;assert torch.count_nonzero(state['embedding_A'][~seen])==0,'Unseen embedding corrections changed';after=errors(state['embedding_A'],state['embedding_B']);states=[]
    with torch.inference_mode():
        for ids,row_ids in zip(val_ids,rows[512:]):
            prediction=forward(state,ids,A,B);stored=forward(state,ids,state['embedding_A'],state['embedding_B']);assert torch.equal(prediction,stored);states.append({'token_ids':row_ids,'ids':ids,'source_embeddings':F.embedding(ids,weight),'hard_embeddings':stored})
    path=out/'embedding.pt';torch.save(state,path);restored=torch.load(path,map_location='cpu',weights_only=True)
    with torch.inference_mode():
        for item in states:assert torch.equal(forward(restored,item['ids'],restored['embedding_A'],restored['embedding_B']),item['hard_embeddings'])
    states_path=out/'validation-embedding-states.pt';torch.save(states,states_path);assert all(tensor_sha(state[n])==digest for n,digest in baseline_hash.items());assert tensor_sha(weight)==original_sha and all(sha(script_dir/n)==digest for n,digest in policy.items()) and not torch.cuda.is_initialized();report={'passed':True,'cpu_only':True,'args':{k:str(v) for k,v in vars(a).items()},'source_model':str(source),'source_index_sha256':sha(index_path),'source_embedding_sha256':original_sha,'source_original_parameters_unchanged':True,'source_sha256':policy,'data':data,'source_blocks':512,'train_token_positions':len(train_ids),'train_unique_token_ids':int(seen.sum()),'unseen_retained_correction_zero':True,'full_embedding_shape':[vocab,k],'actual_optimizer_updates':128,'rank':16,'original_embedding_bytes':weight.numel()*2,'actual_base_bytes':sum(state[n].numel()*state[n].element_size() for n in ('codes','scales','zeros')),'actual_factor_bytes':sum(state[n].numel()*state[n].element_size() for n in ('embedding_A','embedding_B')),'full_vocabulary_rtn_relative_mse':squared_error/denominator,'validation_before':before,'validation_after':after,'events':events,'packed_base_unchanged':True,'stored_training_and_serialized_forward_bitwise_equal':True,'artifact_sha256':sha(path),'validation_states_sha256':sha(states_path),'seconds':time.perf_counter()-began,'limits':'Actual entire source embedding vocabulary encoded at affine4/group128/BF16 metadata, plus fixed128CPU Adam updates of rank16 corrections using only declared512TRAIN token blocks. RTN embedding row packing, not GPTQ on hidden-width activations. Original source and fixed nibble bytes remain unchanged; unseen TRAIN token corrections stay zero;16independent embedding validation blocks, no validation selection. Output head and remaining non-experts unquantized; not whole-model capability or CUDA/single5090 residency/speed proof.'};(out/'summary.json').write_text(json.dumps(report,indent=2));save('completed',passed=True,actual_optimizer_updates=128)

if __name__=='__main__':main()
