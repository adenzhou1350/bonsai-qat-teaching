"""Opt-in fast inference for recorded Q38 step32/96 packed artifacts."""
import argparse
import json
from pathlib import Path
from common import (
    MODES, PERFORMANCE_SCOPE, check_imports, encode, generate, load,
    save_new, verify_artifacts,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--prompt", default="用一句话解释模型量化。")
    parser.add_argument("--mode", choices=MODES, default="decode_rotate")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 64:
        raise ValueError("Use 1..64 new tokens for this short inference entry")
    if args.output is not None and args.output.exists():
        raise FileExistsError("Choose a new output file")
    if not args.verify_only and args.tokenizer is None:
        raise ValueError("--tokenizer is required for generation")
    manifest, identity = verify_artifacts(args.packed, args.tokenizer)
    result = {
        "payload_verified": True, "matrix_files": 402,
        "optimizer_updates": manifest["actual_optimizer_updates"],
        "manifest_SHA256": identity, "source_files_verified": True,
        "tokenizer_verified": args.tokenizer is not None,
        "weights_changed": False, "quality_accepted": False, "GPU_used": False,
    }
    if args.verify_only:
        save_new(args.output, result)
        print(json.dumps(result))
        return
    import torch
    from runtime import set_mode
    model, tokenizer, eos, _ = load(args.packed, args.tokenizer)
    tokens = encode(tokenizer, args.prompt)
    set_mode(model, args.mode)
    generate(model, eos, tokens, 2, collect=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    generated, timings, hashes = generate(model, eos, tokens, args.max_new_tokens)
    check_imports()
    result.update({
        "completed": True, "mode": args.mode, "GPU_used": True,
        "input_token_IDs": tokens, "generated_token_IDs": generated,
        "text": tokenizer.decode(generated, skip_special_tokens=True),
        "ended_with_EOS": generated[-1] in eos,
        "last_position_logit_SHA256": hashes,
        "first_token_seconds": timings[0], "step_seconds": timings,
        "decode_after_first_tokens_per_second":
            (len(timings) - 1) / sum(timings[1:]) if len(timings) > 1 else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "performance_scope": PERFORMANCE_SCOPE,
    })
    save_new(args.output, result)
    print(result["text"])
    print(json.dumps({k: result[k] for k in (
        "mode", "first_token_seconds", "decode_after_first_tokens_per_second",
        "peak_allocated_bytes", "quality_accepted",
    )}))


if __name__ == "__main__":
    main()
