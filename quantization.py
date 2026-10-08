"""Group-wise ternary quantization and identity straight-through gradients."""
import inspect
from types import SimpleNamespace
import torch
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen

@torch.no_grad()
def ternary_grid(weight, group_size=128, iterations=8):
    assert weight.dtype == torch.float32 and weight.shape[-1] % group_size == 0
    groups = weight.reshape(*weight.shape[:-1], -1, group_size)
    scale = groups.abs().mean(-1).clamp_min(1e-8)
    for _ in range(iterations):
        code = (groups / scale[..., None]).round().clamp(-1, 1)
        scale = ((groups * code).sum(-1) / code.square().sum(-1).clamp_min(1)).abs().clamp_min(1e-8)
    scale = scale.bfloat16()
    code = (groups / scale.float()[..., None]).round().clamp(-1, 1).to(torch.int8)
    grid = (code.float() * scale.float()[..., None]).bfloat16().reshape(weight.shape)
    return code.reshape(weight.shape), scale, grid

class StraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, master, grid):
        return grid
    @staticmethod
    def backward(ctx, gradient):
        return gradient.float(), None

def pack_trits(code):
    """Five base-three digits per byte; padding represents zero weights."""
    shape = code.shape; k = shape[-1]
    digits = code.reshape(-1, k).to(torch.int16) + 1
    digits = torch.nn.functional.pad(digits, (0, (-k) % 5), value=1)
    powers = torch.tensor([1, 3, 9, 27, 81], device=code.device)
    return (digits.reshape(-1, digits.shape[-1] // 5, 5) * powers).sum(-1).byte().reshape(*shape[:-1], -1)

def unpack_trits(packed, k):
    powers = torch.tensor([1, 3, 9, 27, 81], device=packed.device)
    digits = packed.to(torch.int16)[..., None] // powers % 3 - 1
    return digits.reshape(*packed.shape[:-1], -1)[..., :k].to(torch.int8)

class QuantizedExperts(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.gate = torch.nn.Parameter(original.gate_up_proj.detach().float())
        self.down = torch.nn.Parameter(original.down_proj.detach().float())
        self.act_fn = original.act_fn
    def forward(self, hidden, indices, probabilities):
        gate = StraightThrough.apply(self.gate, ternary_grid(self.gate)[2])
        down = StraightThrough.apply(self.down, ternary_grid(self.down)[2])
        proxy = SimpleNamespace(num_experts=self.gate.shape[0], gate_up_proj=gate,
                                down_proj=down, act_fn=self.act_fn)
        # Preserve the installed Qwen native routing, activation and index_add.
        return inspect.unwrap(qwen.Qwen3_5MoeExperts.forward)(proxy, hidden, indices, probabilities)
