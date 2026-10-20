"""DRAFT Q38 inference primitives; no GPU correctness or speed claim yet.

Numerical references:
* courses/qwen38_bonsai2/{ternary,rotation}.py in bonsai-qat-teaching.
* The five-trit LUT idea follows experimental122/runtime/
  native122_ternary_lut_decode_v1533.py; no bank routing code is reused.

Contracts:
* Packed rows are uint8, five little-endian base-three digits per byte.
  Digits 0, 1, 2 represent -1, 0, +1. Each row is padded independently.
* Positive finite FP16 scales are per row and per 128 columns. A partial final
  group is supported; it has one scale, just like a complete group.
* The loader must verify canonical bytes (<=242), scales, and +/-1 signs once.
  validate_data() is a synchronous cold-path helper, not a per-layer hot path.
* All public operators are inference-only and require contiguous CUDA inputs.
  CPU inputs are accepted only with TRITON_INTERPRET=1 for bounded controls.
  They validate metadata without synchronizing the GPU. Invalid packed bytes,
  scales, or embedding IDs produce NaN; invalid IDs never address out of bounds.
* embedding() returns the selected rows in the ROTATED basis. The caller must
  run rotate(..., inverse=True) afterwards, matching PackedTernary.forward.
* decode() allocates the full decoded matrix. embedding() allocates only
  ids.numel() * width elements, regardless of vocabulary size.
* LUT creation is lazy and uses 512 bytes/device. Warm it outside CUDA capture.

Hadamard uses tl.gather over the entire block, not a warp-local shuffle.
Every butterfly reads the previous stage's complete logical tensor. Triton
owns the cross-warp lowering/barriers. No handwritten warp-safety assumptions.
The intended Q38 block is 1024 (normalization exactly 1/32); other powers of two
are accepted for controls but still require numerical validation.
"""

import math
import operator
import os
from contextlib import nullcontext

import torch
import triton
import triton.language as tl


GPU_VERIFIED = False
GROUP = 128
_LOOKUPS = {}
_OUTPUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


def _lookup(device):
    """One packed 10-bit LUT entry per input byte; no CUDA work at import."""
    device = torch.device(device)
    key = (device.type, device.index)
    value = _LOOKUPS.get(key)
    if value is None:
        values = [
            sum(((byte // (3 ** digit)) % 3) << (2 * digit)
                for digit in range(5))
            for byte in range(256)
        ]
        value = torch.tensor(values, dtype=torch.int16, device=device)
        _LOOKUPS[key] = value
    return value


def _cuda_contiguous(value, name):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if not value.is_cuda and not (_INTERPRET and value.device.type == "cpu"):
        raise ValueError(f"{name} requires CUDA, or CPU with TRITON_INTERPRET=1")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if value.requires_grad:
        raise ValueError(f"{name}: this draft is inference-only")


def _device_scope(device):
    # The interpreter's CPU path must never enter torch.cuda.device or init CUDA.
    return torch.cuda.device(device) if device.type == "cuda" else nullcontext()


def _packed_metadata(codes, scales, width, dtype):
    width = operator.index(width)
    if width <= 0 or dtype not in _OUTPUT_DTYPES:
        raise ValueError("Use a positive width and FP16/BF16/FP32 output dtype")
    _cuda_contiguous(codes, "codes")
    _cuda_contiguous(scales, "scales")
    if codes.dtype != torch.uint8 or codes.ndim != 2:
        raise ValueError("codes must be a 2D uint8 matrix")
    rows = codes.shape[0]
    packed_width = triton.cdiv(width, 5)
    scale_width = triton.cdiv(width, GROUP)
    if codes.shape[1] != packed_width:
        raise ValueError("Packed rows must have ceil(width/5) bytes")
    if scales.dtype != torch.float16 or tuple(scales.shape) != (rows, scale_width):
        raise ValueError("scales must be FP16 [rows, ceil(width/128)]")
    if codes.device != scales.device:
        raise ValueError("codes and scales must be on the same CUDA device")
    return rows, width, packed_width, scale_width


@triton.jit
def _decoded_values(Codes, Scales, Lookup, row, col, mask,
                    PACKED_WIDTH: tl.constexpr, SCALE_WIDTH: tl.constexpr):
    # int64 row offsets also support matrices whose decoded element count >2^31.
    row64 = row.to(tl.int64)
    byte = tl.load(Codes + row64 * PACKED_WIDTH + col // 5,
                   mask=mask, other=121).to(tl.int32)
    digits = tl.load(Lookup + byte).to(tl.int32)
    trit = ((digits >> (2 * (col % 5))) & 3) - 1
    scale = tl.load(Scales + row64 * SCALE_WIDTH + col // 128,
                    mask=mask, other=1.0).to(tl.float32)
    valid_data = (byte <= 242) & (scale > 0.0) & (scale < float("inf"))
    value = trit.to(tl.float32) * scale
    return tl.where(valid_data, value, float("nan"))


@triton.jit
def _decode_kernel(Codes, Scales, Lookup, Out,
                   ROWS: tl.constexpr, WIDTH: tl.constexpr,
                   PACKED_WIDTH: tl.constexpr, SCALE_WIDTH: tl.constexpr,
                   TILE_ROWS: tl.constexpr, TILE_COLS: tl.constexpr):
    rows = tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)
    cols = tl.program_id(1) * TILE_COLS + tl.arange(0, TILE_COLS)
    mask = (rows[:, None] < ROWS) & (cols[None, :] < WIDTH)
    values = _decoded_values(
        Codes, Scales, Lookup, rows[:, None], cols[None, :], mask,
        PACKED_WIDTH, SCALE_WIDTH)
    tl.store(Out + rows[:, None].to(tl.int64) * WIDTH + cols[None, :],
             values, mask=mask)


@triton.jit
def _embedding_kernel(Codes, Scales, Ids, Lookup, Out,
                      VOCAB: tl.constexpr, WIDTH: tl.constexpr,
                      PACKED_WIDTH: tl.constexpr, SCALE_WIDTH: tl.constexpr,
                      TILE_COLS: tl.constexpr):
    # One program per selected row and column tile; never decode the vocabulary.
    token = tl.program_id(0)
    row = tl.load(Ids + token).to(tl.int64)
    cols = tl.program_id(1) * TILE_COLS + tl.arange(0, TILE_COLS)
    valid_row = (row >= 0) & (row < VOCAB)
    read_mask = valid_row & (cols < WIDTH)
    values = _decoded_values(
        Codes, Scales, Lookup, row, cols, read_mask,
        PACKED_WIDTH, SCALE_WIDTH)
    values = tl.where(valid_row, values, float("nan"))
    tl.store(Out + token.to(tl.int64) * WIDTH + cols, values, mask=cols < WIDTH)


@triton.jit
def _rotate_kernel(X, Signs, Out,
                   WIDTH: tl.constexpr, BLOCK: tl.constexpr,
                   STAGES: tl.constexpr, NORMALIZER: tl.constexpr,
                   INVERSE: tl.constexpr):
    block_id = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    offset = block_id.to(tl.int64) * BLOCK + lane
    sign_offset = (block_id % (WIDTH // BLOCK)) * BLOCK + lane
    value = tl.load(X + offset).to(tl.float32)
    sign = tl.load(Signs + sign_offset).to(tl.float32)
    if not INVERSE:
        value = value * sign

    # Ascending span order is identical to rotation.hadamard. Each gather is
    # from the previous complete stage; XOR never leaves this logical block.
    for stage in tl.static_range(STAGES):
        span = 1 << stage
        peer = tl.gather(value, lane ^ span, axis=0)
        value = tl.where((lane & span) == 0, value + peer, peer - value)

    value = value / NORMALIZER
    if INVERSE:
        value = value * sign
    value = tl.where((sign == 1.0) | (sign == -1.0), value, float("nan"))
    tl.store(Out + offset, value)


def decode(codes, scales, width, dtype=torch.bfloat16):
    """Decode a complete [rows,width] grid directly into FP16, BF16, or FP32."""
    rows, width, packed_width, scale_width = _packed_metadata(
        codes, scales, width, dtype)
    out = torch.empty((rows, width), device=codes.device, dtype=dtype)
    if not rows:
        return out
    with _device_scope(codes.device):
        lookup = _lookup(codes.device)
        _decode_kernel[(triton.cdiv(rows, 4), triton.cdiv(width, 256))](
            codes, scales, lookup, out, rows, width, packed_width, scale_width,
            4, 256, num_warps=4, enable_fp_fusion=False)
    return out


def embedding(codes, scales, ids, width, dtype=torch.bfloat16):
    """Decode only ids' rows, returning [*ids.shape,width] in the rotated basis."""
    vocab, width, packed_width, scale_width = _packed_metadata(
        codes, scales, width, dtype)
    _cuda_contiguous(ids, "ids")
    if ids.dtype not in (torch.int32, torch.int64) or ids.device != codes.device:
        raise ValueError("ids must be int32/int64 on the packed weights' device")
    out = torch.empty((*ids.shape, width), device=codes.device, dtype=dtype)
    if not ids.numel():
        return out
    with _device_scope(codes.device):
        lookup = _lookup(codes.device)
        _embedding_kernel[(ids.numel(), triton.cdiv(width, 256))](
            codes, scales, ids, lookup, out, vocab, width,
            packed_width, scale_width, 256,
            num_warps=4, enable_fp_fusion=False)
    return out


def rotate(x, signs, block=1024, inverse=False):
    """Signed block Hadamard in one kernel, FP32 stages, output x.dtype."""
    _cuda_contiguous(x, "x")
    _cuda_contiguous(signs, "signs")
    block = operator.index(block)
    if x.ndim < 1 or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("x must have a last dimension and FP16/BF16/FP32 dtype")
    width = x.shape[-1]
    if block < 1 or block > 1024 or block & (block - 1) or width % block:
        raise ValueError("Use a power-of-two block <=1024 dividing the width")
    if signs.ndim != 1 or signs.numel() != width or signs.device != x.device:
        raise ValueError("signs must match the full width and CUDA device")
    if signs.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64,
                           torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Unsupported signs dtype")
    out = torch.empty_like(x)
    if not x.numel():
        return out
    with _device_scope(x.device):
        _rotate_kernel[(x.numel() // block,)](
            x, signs, out, width, block, block.bit_length() - 1,
            math.sqrt(block), bool(inverse), num_warps=4, enable_fp_fusion=False)
    return out


def validate_data(codes, scales, signs=None, ids=None):
    """Optional synchronous cold-path validation; works on CPU or CUDA tensors.

    Run when loading untrusted/unverified inputs, not inside each forward call.
    Shape/dtype/device checks remain in the public operators.
    """
    if bool((codes > 242).any()):
        raise ValueError("Noncanonical base-three byte (>242)")
    if not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
        raise ValueError("Scales must be finite and positive")
    if signs is not None and not bool(((signs == 1) | (signs == -1)).all()):
        raise ValueError("Signs must be exactly -1 or +1")
    if ids is not None and (bool((ids < 0).any()) or bool((ids >= codes.shape[0]).any())):
        raise IndexError("Embedding token ID outside the vocabulary")
