"""Four matched rounds on eight fixed development prompts; not a quality benchmark."""
import argparse
import gc
import json
from pathlib import Path
from common import (
    COURSE, PERFORMANCE_SCOPE, PROMPTS_SHA256, check_imports, encode,
    generate, load, sha, verify_artifacts,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output directory")
    manifest, identity = verify_artifacts(args.packed, args.tokenizer)
    prompts = COURSE / "diagnostic-prompts.json"
    if sha(prompts) != PROMPTS_SHA256:
        raise ValueError("Use the unchanged original eight prompts")
    cases = json.loads(prompts.read_text(encoding="utf-8"))
    if len(cases) != 8 or len({r["id"] for r in cases}) != 8 or any(r["max_new_tokens"] != 32 for r in cases):
        raise ValueError("Use the unchanged bounded eight development prompts")
    if args.verify_only:
        print(json.dumps({
            "payload_verified": True, "source_files_verified": True,
            "tokenizer_verified": True, "prompts_verified": True,
            "manifest_SHA256": identity, "GPU_used": False,
        }))
        return
    import torch
    from runtime import set_mode
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "format": "q38-inference-candidate-matched-portable-v1",
        "packed_manifest_SHA256": identity,
        "updates": manifest["actual_optimizer_updates"],
        "prompt_SHA256": sha(prompts), "rounds": [], "passed": False,
        "quality_accepted": False, "weights_changed": False,
        "performance_scope": PERFORMANCE_SCOPE,
        "reserved_memory_note": "Allocator cache is released at round boundaries; identical per-case two-token warmup. Peaks include packed model.",
        "round_order": ["reference", "decode", "decode_rotate", "reference"],
    }

    def save(state):
        report["state"] = state
        temporary = args.output / "progress.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(args.output / "progress.json")

    save("loading_verified_packed_model")
    model, tokenizer, eos, _ = load(args.packed, args.tokenizer)
    prepared = [(case, encode(tokenizer, case["prompt"])) for case in cases]
    for round_id, mode in enumerate(report["round_order"]):
        gc.collect()
        torch.cuda.empty_cache()
        set_mode(model, mode)
        current = {"mode": mode, "cases": []}
        report["rounds"].append(current)
        for index, (case, tokens) in enumerate(prepared):
            save(f"round_{round_id}_{mode}_{case['id']}")
            generate(model, eos, tokens, 2, collect=False)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            generated, timings, hashes = generate(model, eos, tokens, 32)
            row = {
                "id": case["id"], "prompt": case["prompt"], "input_token_IDs": tokens,
                "generated_token_IDs": generated,
                "text": tokenizer.decode(generated, skip_special_tokens=True),
                "ended_with_EOS": generated[-1] in eos,
                "last_position_logit_SHA256": hashes,
                "first_token_seconds": timings[0], "step_seconds": timings,
                "decode_after_first_tokens_per_second":
                    (len(timings) - 1) / sum(timings[1:]) if len(timings) > 1 else None,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            if round_id:
                reference = report["rounds"][0]["cases"][index]
                row["reference_token_IDs_exact"] = generated == reference["generated_token_IDs"]
                row["reference_all_step_logits_bytes_exact"] = hashes == reference["last_position_logit_SHA256"]
            current["cases"].append(row)
            save(f"completed_round_{round_id}_{case['id']}")
            if round_id and not (row["reference_token_IDs_exact"] and row["reference_all_step_logits_bytes_exact"]):
                raise ValueError("Candidate/reference mismatch; preserve report")
            gc.collect()
        elapsed = sum(sum(row["step_seconds"][1:]) for row in current["cases"])
        count = sum(max(0, len(row["generated_token_IDs"]) - 1) for row in current["cases"])
        current["pooled_decode_tokens_per_second"] = count / elapsed if elapsed else None
        save(f"completed_round_{round_id}")
    check_imports()
    report["passed"] = True
    save("matched_reference_and_candidate_completed_review_semantics_separately")
    with (args.output / "summary.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "passed": True, "rounds": len(report["rounds"]), "quality_accepted": False,
        "decode_rates": [r["pooled_decode_tokens_per_second"] for r in report["rounds"]],
    }))


if __name__ == "__main__":
    main()
