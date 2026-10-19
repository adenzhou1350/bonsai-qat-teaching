"""CPU: compare variable-length KL normalization and checkpointed gradients to direct math."""
import json
import torch
from continue_mixed import distillation_objective
torch.set_num_threads(1);torch.manual_seed(381027)
for length in (28,71,128):
    x=torch.randn(1,length,16,requires_grad=True);target=torch.randn_like(x)
    student=torch.nn.Linear(16,23,bias=False);teacher=torch.nn.Linear(16,23,bias=False).requires_grad_(False)
    value=distillation_objective(x,target,student,teacher)
    logits=student(x).float();probabilities=teacher(target).float().softmax(-1)
    expected_KL=(probabilities*(probabilities.log()-logits.log_softmax(-1))).sum()/length
    expected_MSE=(x-target).square().mean()/target.square().mean()
    expected=expected_KL+.1*expected_MSE
    assert torch.allclose(value[0],expected,atol=1e-6,rtol=1e-6)
    got=torch.autograd.grad(value[0],(x,student.weight),retain_graph=True)
    reference=torch.autograd.grad(expected,(x,student.weight))
    assert all(torch.allclose(a,b,atol=1e-6,rtol=1e-6) and torch.isfinite(a).all() for a,b in zip(got,reference))
    with torch.no_grad():actual=distillation_objective(x,target,student,teacher)
    assert all(torch.allclose(a,b,atol=1e-6,rtol=1e-6) for a,b in zip(actual,value))
print(json.dumps({'passed':True,'variable_sequence_lengths':[28,71,128],'KL_normalization_and_checkpointed_gradients_match_direct_math':True,'scope':'tiny CPU fixture; not full27B continuation'}))
