"""Fixed signed block-Hadamard basis; independent PyTorch teaching implementation."""
import hashlib
import math
import torch


def fixed_signs(width, seed=381027):
    """Explicit reproducible signs; not PrismML's unpublished training initialization."""
    digest = hashlib.shake_256(f"bonsai-teaching:{seed}:{width}".encode()).digest(width)
    return torch.tensor([1 if value & 1 else -1 for value in digest], dtype=torch.int8)


def hadamard(x, block=1024):
    if block < 1 or block & (block - 1) or x.shape[-1] % block:
        raise ValueError("A power-of-two block must divide the last dimension")
    shape = x.shape
    z = x.reshape(-1, block)
    span = 1
    while span < block:
        pairs = z.reshape(-1, block // (2 * span), 2, span)
        a, b = pairs.unbind(-2)
        z = torch.stack((a + b, a - b), dim=-2).reshape(-1, block)
        span *= 2
    return (z / math.sqrt(block)).reshape(shape)


def rotate(x, signs, block=1024, inverse=False):
    """Row-vector x R.T; inverse is x R, with R=H S/sqrt(block)."""
    if signs.ndim != 1 or signs.numel() != x.shape[-1]:
        raise ValueError("Explicit signs must match the full input width")
    if not bool(((signs == 1) | (signs == -1)).all()):
        raise ValueError("Signs must be exactly -1 or +1")
    dtype = x.dtype
    if dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    signs = signs.to(device=x.device, dtype=x.dtype)
    result = hadamard(x, block) * signs if inverse else hadamard(x * signs, block)
    return result.to(dtype)
