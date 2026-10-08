"""Full expert-weight ternary QAT, original-teacher KD, export and resume."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as recompute
from model import TextModel
from quantization import ternary_grid,pack_trits,unpack_trits
from prepare_teacher import sha
import checkpoint

def kd_loss(prediction, target, head, chunk=32):
    def part(x,y):
        student=torch.nn.functional.linear(x,head).float()
        with torch.no_grad(): teacher=torch.nn.functional.linear(y,head).float().softmax(-1)
        return torch.nn.functional.kl_div(student.log_softmax(-1),teacher,reduction='sum')
    kl=prediction.new_zeros((),dtype=torch.float32)
    for i in range(0,prediction.shape[1],chunk):
        args=(prediction[:,i:i+chunk],target[:,i:i+chunk])
        kl=kl+(recompute(part,*args,use_reentrant=False) if torch.is_grad_enabled() else part(*args))
    kl=kl/prediction.shape[1]
    mse=(prediction.float()-target.float()).square().mean()/target.float().square().mean().clamp_min(1e-12)
    return kl+.1*mse,kl,mse

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--tokens',required=True)
    p.add_argument('--teacher',required=True); p.add_argument('--output',required=True); p.add_argument('--devices',default='0,1')
    p.add_argument('--steps',type=int,default=4096); p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--checkpoint-every',type=int,default=512); p.add_argument('--resume'); a=p.parse_args()
    out=Path(a.output); out.mkdir(parents=True,exist_ok=False); torch.manual_seed(9917)
    data=torch.load(a.tokens,map_location='cpu',weights_only=True); teacher=Path(a.teacher)
    meta=json.loads((teacher/'metadata.json').read_text())
    assert meta['tokens_sha256']==sha(a.tokens) and meta['model_index_sha256']==sha(Path(a.model)/'model.safetensors.index.json')
    assert 0<a.steps<=len(data['train']) and a.checkpoint_every>0
    targets={}
    for split in ('train','val'):
        row=meta['splits'][split]; assert sha(teacher/f'{split}.bin')==row['sha256']
        assert row['shape']==[len(data[split]),data[split].shape[1]-1,meta['hidden_size']]
        targets[split]=np.memmap(teacher/f'{split}.bin',dtype=np.uint16,mode='r',shape=tuple(row['shape']))
    model=TextModel(a.model,[int(x) for x in a.devices.split(',')],True)
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW([p for _,p in named],lr=a.lr,weight_decay=0,foreach=False)
    identity={'tokens_sha256':meta['tokens_sha256'],'model_index_sha256':meta['model_index_sha256'],
              'teacher_metadata_sha256':sha(teacher/'metadata.json'),'train_order_seed':9117,'group_size':128}
    first=checkpoint.restore(a.resume,named,optimizer,identity) if a.resume else 0
    def target(split,i):
        return torch.from_numpy(targets[split][i].copy()).view(torch.bfloat16)[None].to(model.devices[-1])
    def validate():
        rows=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for i,ids in enumerate(data['val']):
                _,kl,mse=kd_loss(model(ids[:-1][None]),target('val',i),model.head)
                rows.append({'row':i,'kl':float(kl),'mse':float(mse)})
        return rows
    before=validate(); (out/'validation-before.json').write_text(json.dumps(before,indent=2))
    order=torch.randperm(len(data['train']),generator=torch.Generator().manual_seed(9117)).tolist()
    events=[]
    for step in range(first,a.steps):
        i=order[step]; optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss,kl,mse=kd_loss(model(data['train'][i,:-1][None],recompute=True),target('train',i),model.head)
        assert torch.isfinite(loss); loss.backward()
        for _,param in named:
            assert param.grad is not None and param.grad.shape==param.shape
            for low in range(0,len(param),4): assert torch.isfinite(param.grad[low:low+4]).all()
        norm=torch.nn.utils.clip_grad_norm_([p for _,p in named],1.,foreach=False)
        assert torch.isfinite(norm); optimizer.step()
        event={'step':step+1,'train_row':i,'kl':float(kl.detach()),'mse':float(mse.detach())}
        events.append(event); print(json.dumps(event),flush=True)
        with (out/'train.jsonl').open('a') as f: f.write(json.dumps(event)+'\n')
        del loss,kl,mse
        if (step+1)%a.checkpoint_every==0 or step+1==a.steps:
            optimizer.zero_grad(set_to_none=True)
            checkpoint.save(out/f'checkpoint-{step+1:06d}',named,optimizer,step+1,identity)
    after=validate(); (out/'validation-after.json').write_text(json.dumps(after,indent=2))
    packed=out/'packed'; packed.mkdir(); matrices=[]
    for name,param in named:
        code,scale,grid=ternary_grid(param); codes=pack_trits(code)
        restored=(unpack_trits(codes,param.shape[-1]).reshape(*scale.shape,128).float()*scale.float()[...,None]).bfloat16().reshape(param.shape)
        assert torch.equal(restored.view(torch.int16),grid.view(torch.int16))
        file=packed/(name+'.pt'); torch.save({'shape':list(param.shape),'codes':codes.cpu(),'scales':scale.cpu(),'group_size':128},file)
        matrices.append({'name':name,'file':file.name,'sha256':sha(file)})
        del code,scale,grid,codes,restored
    (packed/'manifest.json').write_text(json.dumps({'identity':identity,'actual_updates':a.steps,'matrices':matrices,
        'retained_weights':'Load unchanged non-expert text weights from the source model.',
        'independent_capability_accepted':False},indent=2))
if __name__=='__main__': main()
