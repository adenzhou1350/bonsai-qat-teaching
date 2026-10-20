"""Opt-in inference experiments; preserve the frozen packed reference and weights.

No training/export behavior is changed. A candidate remains unqualified until
operator checks and matched full-model measurements on an admitted GPU pass.
"""
import torch
import torch.nn.functional as F

import kernels
from ternary import PackedTernary
from rotation import rotate as reference_rotate

MODES = ("reference", "decode", "decode_rotate")


class CandidateTernary(torch.nn.Module):
    def __init__(self, reference):
        super().__init__()
        if not isinstance(reference, PackedTernary):
            raise TypeError("Expected the validated teaching PackedTernary")
        if reference.group != 128 or reference.block != 1024:
            raise ValueError("This candidate covers only the recorded g128/block1024")
        if reference.width % 1024:
            raise ValueError("The recorded rotation requires complete 1024 blocks")
        # Ownership of the same tensors moves under this wrapper; no weight copy.
        self.reference = reference
        self.mode = "reference"

    def forward(self, x):
        r = self.reference
        if self.mode == "reference":
            return r(x)
        if torch.is_grad_enabled():
            raise RuntimeError("Candidate kernels are inference-only")
        rotate = kernels.rotate if self.mode == "decode_rotate" else reference_rotate
        if r.embedding:
            selected = kernels.embedding(r.codes, r.scales, x.contiguous(), r.width, r.compute_dtype)
            return rotate(selected, r.signs, r.block, inverse=True)
        # Keep exactly the reference BF16 weight rounding and F.linear reduction.
        weight = kernels.decode(r.codes, r.scales, r.width, x.dtype)
        activation = x.contiguous() if self.mode == "decode_rotate" else x
        return F.linear(rotate(activation, r.signs, r.block), weight)


def install(model):
    """Call after the original identity-checking loader; initially reference mode."""
    if any(isinstance(m, CandidateTernary) for m in model.modules()):
        raise ValueError("A candidate is already installed")
    found = [(n, m) for n, m in model.named_modules() if isinstance(m, PackedTernary)]
    if not found:
        raise ValueError("No validated packed modules")
    for name, module in found:
        # Cold-path checks remain outside measured generation; only static tensors.
        kernels.validate_data(module.codes, module.scales, signs=module.signs)
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, leaf, CandidateTernary(module))
    return len(found)


def set_mode(model, mode):
    if mode not in MODES:
        raise ValueError("Unknown candidate mode")
    count = 0
    for module in model.modules():
        if isinstance(module, CandidateTernary):
            module.mode = mode
            count += 1
    if not count:
        raise ValueError("Install the candidate before selecting a mode")
    return count
