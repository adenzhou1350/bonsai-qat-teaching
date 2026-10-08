"""One bank at a time; publish and verify full FP32 optimizer checkpoints."""
import json,os
from pathlib import Path
import torch
from prepare_teacher import sha

def save(path, named, optimizer, step, identity):
    path=Path(path); temp=path.with_suffix('.partial'); temp.mkdir(exist_ok=False); rows=[]
    for i,(name,param) in enumerate(named):
        state=optimizer.state[param]
        value={'name':name,'master':param.detach().to('cpu',copy=True),
               **{key:t.detach().to('cpu',copy=True) for key,t in state.items()}}
        file=temp/f'{i:03d}.pt'; torch.save(value,file)
        with file.open('rb') as f: os.fsync(f.fileno())
        rows.append({'file':file.name,'name':name,'sha256':sha(file)}); del value
    torch.save({'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},temp/'rng.pt')
    manifest={'step':step,'identity':identity,'banks':rows,'rng_sha256':sha(temp/'rng.pt'),
              'optimizer':{k:v for k,v in optimizer.param_groups[0].items() if k!='params'}}
    (temp/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for name in ('rng.pt','manifest.json'):
        with (temp/name).open('rb') as f: os.fsync(f.fileno())
    fd=os.open(temp,os.O_RDONLY); os.fsync(fd); os.close(fd); temp.rename(path)
    fd=os.open(path.parent,os.O_RDONLY); os.fsync(fd); os.close(fd)

def restore(path, named, optimizer, identity):
    path=Path(path); v=json.loads((path/'manifest.json').read_text())
    assert v['identity']==identity and [r['name'] for r in v['banks']]==[n for n,_ in named]
    assert all(sha(path/r['file'])==r['sha256'] for r in v['banks']) and sha(path/'rng.pt')==v['rng_sha256']
    with torch.no_grad():
        for row,(name,param) in zip(v['banks'],named):
            state=torch.load(path/row['file'],map_location='cpu',weights_only=True)
            assert state['name']==name and state['master'].shape==param.shape and state['master'].dtype==torch.float32
            param.copy_(state['master']); param.grad=None
            optimizer.state[param]={'step':state['step'],'exp_avg':state['exp_avg'].to(param.device),
                                    'exp_avg_sq':state['exp_avg_sq'].to(param.device)}
            del state
    optimizer.param_groups[0].update(v['optimizer'])
    rng=torch.load(path/'rng.pt',map_location='cpu',weights_only=True)
    torch.set_rng_state(rng['cpu']); torch.cuda.set_rng_state_all(rng['cuda'])
    return v['step']
