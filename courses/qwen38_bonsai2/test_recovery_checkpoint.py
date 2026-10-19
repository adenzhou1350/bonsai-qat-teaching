"""Linux CPU: lossless master/Adafactor/RNG resume and corruption/budget rejection."""
import json,random,tempfile
from pathlib import Path
import torch
from transformers.optimization import Adafactor
from recovery_checkpoint import load_checkpoint,save_checkpoint

torch.set_num_threads(1);torch.manual_seed(381027);random.seed(381027)
def make():
    model=torch.nn.Linear(16,32,bias=False)
    optimizer=Adafactor(model.parameters(),lr=1e-5,relative_step=False,scale_parameter=False,
                       warmup_init=False,beta1=None,clip_threshold=1.)
    return model,optimizer
def step(model,optimizer):
    optimizer.zero_grad(set_to_none=True);x=torch.randn(4,16)
    loss=model(x).square().mean()*random.random();loss.backward();optimizer.step();return float(loss.detach())
metadata={'dataset_SHA256':'fixture-not-real-data','step':1}
with tempfile.TemporaryDirectory() as base:
    base=Path(base);a,opt=make();step(a,opt)
    manifest=save_checkpoint(a,opt,base/'saved',metadata,maximum_bytes=1000000,reserve_bytes=0)
    expected_loss=step(a,opt);expected=a.weight.detach().clone()
    b,resumed=make();load_checkpoint(b,resumed,base/'saved',metadata)
    actual_loss=step(b,resumed)
    assert actual_loss==expected_loss and torch.equal(expected,b.weight)
    for key,value in next(iter(opt.state.values())).items():
        other=next(iter(resumed.state.values()))[key]
        assert torch.equal(value,other) if torch.is_tensor(value) else value==other
    path=base/'saved'/manifest['masters'][0]['file'];raw=bytearray(path.read_bytes());raw[len(raw)//2]^=1;path.write_bytes(raw)
    unchanged=b.weight.detach().clone()
    try:load_checkpoint(b,resumed,base/'saved',metadata)
    except ValueError:pass
    else:raise AssertionError('Corrupted compressed master accepted')
    assert torch.equal(unchanged,b.weight)
    try:save_checkpoint(b,resumed,base/'over-budget',metadata,maximum_bytes=1,reserve_bytes=0)
    except RuntimeError:pass
    else:raise AssertionError('Storage budget ignored')
    assert not (base/'over-budget/manifest.json').exists()
    c,ada=make();initial=c.weight.detach().clone();step(c,ada)
    xor_meta={'dataset_SHA256':'xor-fixture-not-real-data','step':1}
    xor=save_checkpoint(c,ada,base/'xor',xor_meta,maximum_bytes=1000000,reserve_bytes=0,
                        base_factory=lambda name,param:initial)
    next_loss=step(c,ada);next_weight=c.weight.detach().clone()
    d,restored=make()
    try:load_checkpoint(d,restored,base/'xor',xor_meta)
    except ValueError:pass
    else:raise AssertionError('Wrong original master accepted')
    with torch.no_grad():d.weight.copy_(initial)
    load_checkpoint(d,restored,base/'xor',xor_meta)
    assert next_loss==step(d,restored) and torch.equal(next_weight,d.weight)
print(json.dumps({'passed':True,'next_loss_masters_and_Adafactor_state_bitwise_equal_after_disk_restore':True,
                  'corruption_rejected_before_model_mutation':True,'incomplete_over_budget_checkpoint_rejected':True,
                  'lossless_XOR_checkpoint_next_update_equal_and_wrong_base_rejected':True,
                  'scope':'small Linux CPU fixture, not full27B resume execution'}))
