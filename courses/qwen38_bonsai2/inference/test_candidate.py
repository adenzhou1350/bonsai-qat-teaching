"""Small CPU-interpreter/GPU operator checks, not model performance evidence."""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import kernels
from rotation import fixed_signs, rotate as reference_rotate
from ternary import pack, unpack, unpack_grid, PackedTernary
from runtime import CandidateTernary


def run(device):
    torch.set_num_threads(2)
    torch.manual_seed(20261020)
    if device == "cpu":
        if os.environ.get("TRITON_INTERPRET") != "1":
            raise ValueError("CPU checks must use the Triton interpreter")
        assert not torch.cuda.is_initialized()
    results = []
    # Triton3.6 CPU interpreter's implicit FP32->BF16 cast truncates; its
    # explicit RTNE branch also lacks tie-to-even. Do not weaken CUDA checks.
    # Interpreter BF16 store is explicitly OUT OF SCOPE, not a passing test.
    dtypes = (torch.float32, torch.float16) if device == "cpu" else (
        torch.float32, torch.float16, torch.bfloat16)

    def record(name, actual, expected):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype, name
        exact = torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
        if not exact:
            error = float((actual.float() - expected.float()).abs().max())
            raise AssertionError(f"{name}: max absolute error {error}")
        results.append({"case": name, "exact": True, "elements": actual.numel()})

    # Every legal packed byte, including padding at width128. Both signs and zero.
    all_bytes = torch.arange(243, dtype=torch.uint8, device=device)[:, None].repeat(1, 26)
    scales = torch.linspace(2**-14, 2, 243, dtype=torch.float16, device=device)[:, None]
    kernels.validate_data(all_bytes, scales)
    for dtype in dtypes:
        expected = unpack_grid(unpack(all_bytes, 128), scales).to(dtype)
        record(f"all_243_bytes_{dtype}", kernels.decode(all_bytes, scales, 128, dtype), expected)
    if device == "cpu":
        record("decode_FP32_then_Torch_BF16_NOT_Triton_BF16_store",
               kernels.decode(all_bytes, scales, 128, torch.float32).bfloat16(),
               unpack_grid(unpack(all_bytes, 128), scales).bfloat16())

    for width in (128, 1024, 5120, 17408):
        # Three rows also exercise row tiling and non-multiple-of-five widths.
        code = torch.randint(-1, 2, (3, width), device=device, dtype=torch.int8)
        packed = pack(code)
        scales = (torch.rand((3, width // 128), device=device) + 2**-12).half()
        reference = unpack_grid(code, scales)
        ids = torch.tensor([[2, 0, 2], [1, 2, 0]], device=device)
        for dtype in dtypes:
            decoded = kernels.decode(packed, scales, width, dtype)
            record(f"decode_w{width}_{dtype}", decoded, reference.to(dtype))
            selected = kernels.embedding(packed, scales, ids, width, dtype)
            record(f"embedding_w{width}_{dtype}", selected, reference.to(dtype)[ids])
    # Scale extremes and BF16 rounding ties (CUDA only for the BF16 store).
    scale_values = torch.tensor([2**-24, 2**-14, 1.00390625, 1.0048828125,
                                 1.0029296875, 65504], dtype=torch.float16, device=device)
    edge_code = torch.arange(128, device=device).remainder(3).sub(1).to(torch.int8)[None].repeat(6, 1)
    edge_scales = scale_values[:, None]
    for dtype in dtypes:
        record(f"scale_extremes_and_rounding_ties_{dtype}",
               kernels.decode(pack(edge_code), edge_scales, 128, dtype),
               unpack_grid(edge_code, edge_scales).to(dtype))
    for width in (1, 129):
        partial_code = torch.randint(-1, 2, (3, width), device=device, dtype=torch.int8)
        partial_scale = torch.ones((3, (width + 127) // 128), dtype=torch.float16, device=device)
        for dtype in dtypes:
            record(f"partial_group_w{width}_{dtype}",
                   kernels.decode(pack(partial_code), partial_scale, width, dtype),
                   partial_code.to(dtype))
        empty_ids = torch.empty((0,), dtype=torch.int32, device=device)
        record(f"empty_embedding_w{width}",
               kernels.embedding(pack(partial_code), partial_scale, empty_ids, width, torch.float32),
               torch.empty((0, width), dtype=torch.float32, device=device))
        ids32 = torch.tensor([2, 0], device=device, dtype=torch.int32)
        record(f"int32_embedding_w{width}",
               kernels.embedding(pack(partial_code), partial_scale, ids32, width, torch.float32),
               partial_code[ids32.long()].float())

    for width in (1024, 5120, 17408):
        signs = fixed_signs(width).to(device)
        for dtype in dtypes:
            x = torch.randn((2, width), device=device).to(dtype)
            for inverse in (False, True):
                record(f"rotation_w{width}_{dtype}_inverse{inverse}",
                       kernels.rotate(x, signs, inverse=inverse),
                       reference_rotate(x, signs, inverse=inverse))

    # End-to-end small linear and embedding wrappers, both decode modes.
    width = 1024
    for embedding in (False, True):
        code = torch.randint(-1, 2, (9, width), dtype=torch.int8, device=device)
        value = {"format": "teaching-rotated-ternary-v1", "codes": pack(code),
                 "scales": (torch.rand((9, 8), device=device) + 1e-3).half(),
                 "shape": [9, width], "group": 128, "block": 1024,
                 "signs": fixed_signs(width).to(device), "embedding": embedding,
                 "compute_dtype": "float32" if device == "cpu" else "bfloat16"}
        reference = PackedTernary(value)
        candidate = CandidateTernary(reference)
        x = torch.tensor([[0, 8, 4]], device=device) if embedding else torch.randn(2, width, device=device).to(
            torch.float32 if device == "cpu" else torch.bfloat16)
        with torch.inference_mode():
            expected = reference(x)
            for mode in ("decode", "decode_rotate"):
                candidate.mode = mode
                record(f"module_embedding{embedding}_{mode}", candidate(x), expected)
            noncontiguous = (torch.tensor([[0, 1, 2, 3, 4, 5]], device=device)[:, ::2] if embedding
                             else torch.randn(2, 3, width * 2, device=device).to(x.dtype)[..., ::2])
            assert not noncontiguous.is_contiguous()
            expected = reference(noncontiguous)
            for mode in ("decode", "decode_rotate"):
                candidate.mode = mode
                record(f"module_noncontiguous_embedding{embedding}_{mode}",
                       candidate(noncontiguous), expected)

    # Data admission is still required: bad codes, scales, and signs are rejected.
    bads = [
        (torch.full((1, 26), 243, dtype=torch.uint8, device=device), torch.ones(1, 1, dtype=torch.float16, device=device), None),
        (torch.full((1, 26), 255, dtype=torch.uint8, device=device), torch.ones(1, 1, dtype=torch.float16, device=device), None),
        (all_bytes[:1], torch.zeros(1, 1, dtype=torch.float16, device=device), None),
        (all_bytes[:1], torch.full((1, 1), float("nan"), dtype=torch.float16, device=device), None),
        (all_bytes[:1], torch.full((1, 1), float("inf"), dtype=torch.float16, device=device), None),
        (all_bytes[:1], -torch.ones(1, 1, dtype=torch.float16, device=device), None),
        (all_bytes[:1], scales[:1, :1], torch.zeros(128, dtype=torch.int8, device=device)),
    ]
    for i, (packed, scale, signs) in enumerate(bads):
        try:
            kernels.validate_data(packed, scale, signs=signs)
        except (ValueError, TypeError):
            results.append({"case": f"invalid_payload_{i}", "rejected": True})
        else:
            raise AssertionError("Invalid packed data was accepted")
    if device == "cpu":
        assert not torch.cuda.is_initialized()
    return {"passed": True, "device": device, "interpreter": os.environ.get("TRITON_INTERPRET") == "1",
            "BF16_kernel_store_verified": device == "cuda",
            "unsupported_interpreter_dtypes": ["bfloat16"] if device == "cpu" else [],
            "cases": results, "CUDA_initialized": torch.cuda.is_initialized(),
            "full_model_quality_or_speed_claimed": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Preserve earlier evidence")
    start = time.monotonic()
    result = run(a.device)
    result["seconds"] = time.monotonic() - start
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "cases": len(result["cases"]),
                      "seconds": result["seconds"], "GPU_used": a.device == "cuda"}))
