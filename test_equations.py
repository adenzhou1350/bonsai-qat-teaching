"""Small CPU checks of equations, gradients, packing and restart semantics."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from quantization import ternary_grid,StraightThrough,pack_trits,unpack_trits,QuantizedExperts
from train import kd_loss
import checkpoint

def main():
    torch.manual_seed(31)
    master=torch.randn(4,256,128,requires_grad=True)
    code,scale,grid=ternary_grid(master)
    assert torch.equal(unpack_trits(pack_trits(code),128),code)
    assert set(code.unique().tolist())<= {-1,0,1} and scale.dtype==torch.bfloat16
    incoming=torch.randn_like(grid)
    StraightThrough.apply(master,grid).backward(incoming)
    assert torch.equal(master.grad,incoming.float())
    odd=torch.randint(-1,2,(7,131),dtype=torch.int8)
    assert torch.equal(unpack_trits(pack_trits(odd),131),odd)
    experts=QuantizedExperts(SimpleNamespace(gate_up_proj=master.detach(),
        down_proj=torch.randn(4,128,128),act_fn=torch.nn.functional.silu))
    hidden=torch.randn(5,128,dtype=torch.bfloat16)
    route=torch.tensor([[0,1],[2,3],[0,2],[1,3],[1,2]])
    weights=torch.ones(5,2,dtype=torch.bfloat16)/2
    output=experts(hidden,route,weights); output.float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in experts.parameters())
    x=torch.randn(1,11,6,requires_grad=True); y=torch.randn_like(x); head=torch.randn(17,6)
    loss,kl,mse=kd_loss(x,y,head,chunk=4)
    full=torch.nn.functional.kl_div(torch.nn.functional.linear(x,head).log_softmax(-1),
        torch.nn.functional.linear(y,head).softmax(-1),reduction='sum')/11
    assert torch.allclose(kl,full,atol=1e-6)
    actual=torch.autograd.grad(loss,x,retain_graph=True)[0]
    expected=torch.autograd.grad(full+.1*mse,x)[0]
    assert torch.allclose(actual,expected,atol=1e-6)
    parameter=torch.nn.Parameter(torch.randn(3,4)); optimizer=torch.optim.AdamW([parameter],lr=1e-4,weight_decay=0,foreach=False)
    def update(p,o):
        o.zero_grad(set_to_none=True); p.square().sum().backward(); o.step()
    update(parameter,optimizer)
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'checkpoint'; identity={'fixture':'CPU_equations_only'}
        checkpoint.save(path,[('bank',parameter)],optimizer,1,identity)
        update(parameter,optimizer); expected=parameter.detach().clone()
        fresh=torch.nn.Parameter(torch.zeros_like(parameter)); fresh_optimizer=torch.optim.AdamW([fresh],lr=.2,foreach=False)
        assert checkpoint.restore(path,[('bank',fresh)],fresh_optimizer,identity)==1
        update(fresh,fresh_optimizer); assert torch.equal(fresh,expected)
        for key in ('step','exp_avg','exp_avg_sq'): assert torch.equal(optimizer.state[parameter][key],fresh_optimizer.state[fresh][key])
        with (path/'000.pt').open('ab') as f: f.write(b'corrupt')
        before=fresh.detach().clone()
        try: checkpoint.restore(path,[('bank',fresh)],fresh_optimizer,identity)
        except AssertionError: pass
        else: raise AssertionError('Corrupt checkpoint accepted')
        assert torch.equal(fresh,before)
    print('PASS: ternary grid, identity STE, packing, native expert gradients, full-position KL gradients, checkpoint next-step equivalence, corruption rejection')
if __name__=='__main__': main()
