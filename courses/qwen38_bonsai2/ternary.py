"""Rotated g128 ternary QAT with FP16 scales, including inverse-basis embeddings."""
import math
import torch
from rotation import rotate, fixed_signs


@torch.no_grad()
def quantize(master, group=128, iterations=8):
    if master.dtype != torch.float32 or master.ndim != 2 or master.shape[1] % group:
        raise ValueError("Expected a two-dimensional FP32 master with complete groups")
    if group < 1 or iterations < 1 or not bool(torch.isfinite(master).all()):
        raise ValueError("Invalid group/iterations or non-finite master")
    groups = master.reshape(master.shape[0], -1, group)
    scale = groups.abs().mean(-1).clamp_min(1e-8)
    for _ in range(iterations):
        code = (groups / scale[..., None]).round().clamp(-1, 1)
        scale = ((groups * code).sum(-1) / code.square().sum(-1).clamp_min(1)).abs().clamp_min(1e-8)
    scale = scale.to(torch.float16).clamp_min(torch.finfo(torch.float16).tiny)
    if not bool(torch.isfinite(scale).all()):
        raise ValueError("FP16 scale overflow")
    code = (groups / scale.float()[..., None]).round().clamp(-1, 1).to(torch.int8)
    return code.reshape_as(master), scale


def unpack_grid(code, scales, group=128):
    return (code.reshape(*scales.shape, group).float() * scales.float()[..., None]).reshape_as(code)


def pack(code):
    if code.dtype != torch.int8 or code.ndim != 2 or not bool(((code >= -1) & (code <= 1)).all()):
        raise ValueError("Expected two-dimensional int8 ternary codes")
    digits = torch.nn.functional.pad(code.to(torch.int16) + 1, (0, (-code.shape[1]) % 5), value=1)
    powers = torch.tensor([1, 3, 9, 27, 81], device=code.device)
    return (digits.reshape(code.shape[0], -1, 5) * powers).sum(-1).byte()


def unpack(packed, width):
    if packed.dtype != torch.uint8 or packed.ndim != 2 or packed.shape[1] != math.ceil(width / 5):
        raise ValueError("Invalid packed shape or dtype")
    if bool((packed > 242).any()):
        raise ValueError("Invalid base-three byte")
    powers = torch.tensor([1, 3, 9, 27, 81], device=packed.device)
    digits = packed.to(torch.int16)[..., None] // powers % 3 - 1
    return digits.reshape(packed.shape[0], -1)[:, :width].to(torch.int8)


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, master, grid):
        return grid

    @staticmethod
    def backward(ctx, gradient):
        return gradient.float(), None


class RotatedTernary(torch.nn.Module):
    def __init__(self, original, embedding=False, block=1024, group=128, seed=381027):
        super().__init__()
        if original.ndim != 2:
            raise ValueError("Expected a matrix")
        self.embedding, self.block, self.group = embedding, block, group
        self.register_buffer("signs", fixed_signs(original.shape[1], seed).to(original.device))
        self.master = torch.nn.Parameter(rotate(original.detach().float(), self.signs, block))

    def forward(self, x):
        code, scales = quantize(self.master, self.group)
        weight = STE.apply(self.master, unpack_grid(code, scales, self.group))
        if self.embedding:
            return rotate(torch.nn.functional.embedding(x, weight), self.signs, self.block, inverse=True)
        return torch.nn.functional.linear(rotate(x, self.signs, self.block), weight.to(x.dtype))

    @torch.no_grad()
    def packed(self):
        code, scales = quantize(self.master, self.group)
        return {"format": "teaching-rotated-ternary-v1", "codes": pack(code).cpu(),
                "scales": scales.cpu(), "shape": list(self.master.shape),
                "signs": self.signs.cpu(), "block": self.block, "group": self.group,
                "embedding": self.embedding}


class PackedTernary(torch.nn.Module):
    """Portable correctness reference; expands weights and is not a fused GPU kernel."""
    def __init__(self, value):
        super().__init__()
        if value["format"] != "teaching-rotated-ternary-v1":
            raise ValueError("This loader accepts only the teaching format")
        self.block, self.group, self.embedding = value["block"], value["group"], value["embedding"]
        self.compute_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                              "float16": torch.float16}[value.get("compute_dtype", "float32")]
        shape = value["shape"]
        code = unpack(value["codes"], shape[1])
        if list(code.shape) != shape or list(value["scales"].shape) != [shape[0], shape[1] // self.group]:
            raise ValueError("Packed matrix/scales do not match declared shape")
        if value["scales"].dtype != torch.float16 or not bool(torch.isfinite(value["scales"]).all()) or bool((value["scales"] <= 0).any()):
            raise ValueError("Invalid FP16 scales")
        self.register_buffer("codes", value["codes"])
        self.register_buffer("scales", value["scales"])
        self.register_buffer("signs", value["signs"])
        self.width = shape[1]

    def forward(self, x):
        weight = unpack_grid(unpack(self.codes, self.width), self.scales, self.group)
        if self.embedding:
            return rotate(torch.nn.functional.embedding(x, weight.to(self.compute_dtype)), self.signs, self.block, inverse=True)
        return torch.nn.functional.linear(rotate(x, self.signs, self.block), weight.to(x.dtype))
