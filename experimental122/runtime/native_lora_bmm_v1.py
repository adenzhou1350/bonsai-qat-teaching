import hashlib
from pathlib import Path
import torch
def projection_variant(bank, inputs, ids, mode):
    routes = len(ids)
    _, n, k = bank.shape
    weights = bank.weights(ids)
    base = torch.empty((routes, n), device='cuda', dtype=torch.bfloat16)
    if mode == 'lora_only':
        for route in range(routes):
            torch.mm(inputs[route:route + 1], weights[route].transpose(-2, -1), out=base[route:route + 1])
    else:
        torch.bmm(inputs[:, None, :], weights.transpose(-2, -1), out=base[:, None, :])
    del weights
    a = bank.lora_a[ids]
    b = bank.lora_b[ids]
    low = torch.empty((routes, 8), device='cuda', dtype=torch.bfloat16)
    if mode == 'base_only':
        for route in range(routes):
            torch.mm(inputs[route:route + 1], a[route].transpose(-2, -1), out=low[route:route + 1])
    else:
        torch.bmm(inputs[:, None, :], a.transpose(-2, -1), out=low[:, None, :])
    del a
    correction = torch.empty_like(base)
    if mode == 'base_only':
        for route in range(routes):
            torch.mm(low[route:route + 1], b[route].transpose(-2, -1), out=correction[route:route + 1])
    else:
        torch.bmm(low[:, None, :], b.transpose(-2, -1), out=correction[:, None, :])
    return (base + correction * bank.lora_scale).to(torch.bfloat16)
def candidate(bank,inputs,ids):return projection_variant(bank,inputs,ids,'lora_only')
def install():
 import small_batch122_experts_v2 as module
 assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()=='7020398cc0538e8b23480cccd6c26d0047caaaa85b79bcf5677be96aa9500f3c'
 original=module.projection;module.projection=candidate;return module,original
