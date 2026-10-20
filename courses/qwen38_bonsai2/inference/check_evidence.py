"""Check the recorded Q38 inference evidence using only Python's standard library."""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import sys


COURSE = "courses/qwen38_bonsai2"
EVIDENCE = "docs/experiments/2026-10-20-Q38-inference-v1.json"
ORDER = ["reference", "decode", "decode_rotate", "reference"]
PACKED_SHA256 = "c53fa86911cd61f85321d1305fcab03c39a45c38fa5f2fc18d4cdb982988eed9"
PROMPTS_SHA256 = "62a7f9bc8a2627013b6553d97628c9018df2f07e983a0648e6250dc421548ba3"
SOURCE_NAMES = {
    "control.py", "offload.py", "policy.py", "rotation.py", "ternary.py",
    "infer_prompt.py", "diagnostic-prompts.json", "inference/benchmark.py",
    "inference/common.py", "inference/infer.py", "inference/kernels.py",
    "inference/runtime.py", "inference/test_candidate.py",
    "inference/test_operators.py", "inference/tokenizer-files.json",
}
NUMERICAL_NAMES = {
    "kernels.py", "runtime.py", "test_candidate.py", "control.py",
    "offload.py", "policy.py", "ternary.py", "rotation.py",
}
ORIGINAL_SUMMARY_KEYS = (
    "format", "packed_manifest_SHA256", "updates", "prompt_SHA256", "rounds",
    "passed", "quality_accepted", "weights_changed", "performance_scope",
    "reserved_memory_note", "round_order", "state",
)


class EvidenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def safe_file(root, name):
    require(isinstance(name, str) and "\\" not in name and ":" not in name,
            "Unsafe public path")
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and name == relative.as_posix()
            and ".." not in relative.parts, "Unsafe public path")
    root = Path(root).resolve()
    path = (root / name).resolve()
    require(path.is_relative_to(root) and path.is_file(),
            "Missing or escaped public file: " + name)
    return path


def read_json(root, name):
    return json.loads(safe_file(root, name).read_text(encoding="utf-8"))


def is_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def lf_sha(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check_sources(root, data):
    """Runtime source identity is raw-byte identity, not canonical-LF provenance."""
    course = Path(root) / COURSE
    sources = read_json(course, "inference/source-files.json")
    require(set(sources) == SOURCE_NAMES, "Source file coverage changed")
    attributes = safe_file(root, ".gitattributes").read_text(encoding="utf-8")
    preserved = COURSE + "/inference/tokenizer-files.json"
    rules = [line.split() for line in attributes.splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    require(any(rule[0] == preserved and "-text" in rule and "-eol" in rule
                for rule in rules), "Missing raw-byte tokenizer manifest attribute")
    for name, expected in sources.items():
        require(is_sha(expected), "Invalid source SHA256: " + name)
        raw = safe_file(course, name).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected,
                "Source SHA256 mismatch: " + name)
        if name != "inference/tokenizer-files.json":
            require(b"\r\n" not in raw, "Expected LF source bytes: " + name)
    frozen = data["provenance"]["frozen_numerical_source_SHA256"]
    require(set(frozen) == NUMERICAL_NAMES, "Numerical provenance coverage changed")
    for name, expected in frozen.items():
        relative = "inference/" + name if name in {
            "kernels.py", "runtime.py", "test_candidate.py"} else name
        require(expected == sources[relative], "Numerical provenance mismatch: " + name)
    require(data["provenance"]["public_reproduction_entry"]
            == COURSE + "/inference/benchmark.py", "Wrong public benchmark entry")
    require(sources["diagnostic-prompts.json"] == PROMPTS_SHA256,
            "Development prompt source changed")
    return sources


def check_public_provenance(root, data):
    """Repository provenance uses canonical LF, independently of runtime raw hashes."""
    provenance = read_json(root, COURSE + "/inference/provenance.json")
    require(provenance["format"] == "q38-fast-inference-provenance-v1"
            and provenance["record"] == EVIDENCE, "Wrong public provenance record")
    files = provenance["file_sha256"]
    required = {COURSE + "/" + name for name in SOURCE_NAMES
                if name.startswith("inference/")}
    required |= {COURSE + "/inference/" + name for name in (
        "source-files.json", "check_evidence.py", "test_evidence.py")}
    require(required <= set(files), "Public provenance is missing inference source files")
    for name, expected in files.items():
        require(is_sha(expected) and lf_sha(safe_file(root, name)) == expected,
                "Public provenance SHA256 mismatch: " + name)
    require(provenance["numerical_sources_identical_to_measured_v1"]
            == ["kernels.py", "runtime.py", "test_candidate.py"],
            "Wrong measured numerical source mapping")
    for name in provenance["numerical_sources_identical_to_measured_v1"]:
        require(files[COURSE + "/inference/" + name]
                == data["provenance"]["frozen_numerical_source_SHA256"][name],
                "Measured/public source mismatch: " + name)
    weight = provenance["validated_weights"]["step96"]
    require(weight["manifest_SHA256"] == PACKED_SHA256
            and weight["original_driver_four_rounds_passed"] is True
            and weight["published_weights"] is False, "Wrong measured weight provenance")
    require(provenance["quality_accepted"] is False
            and provenance["new_training"] is False
            and provenance["weights_changed"] is False,
            "Unsupported public provenance claim")


def positive(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            "Non-positive or non-finite " + name)


def close(actual, expected, name):
    require(type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
            "Arithmetic mismatch: " + name)


def tokens(value, name, maximum):
    require(isinstance(value, list) and 1 <= len(value) <= maximum
            and all(type(v) is int and v >= 0 for v in value), "Invalid " + name)


def check_operators(data):
    operators = data["operator_checks"]
    require(operators["passed"] is True and operators["device"] == "cuda"
            and operators["interpreter"] is False
            and operators["BF16_kernel_store_verified"] is True
            and operators["CUDA_initialized"] is True
            and operators["full_model_quality_or_speed_claimed"] is False
            and operators["unsupported_interpreter_dtypes"] == [],
            "Unsupported operator pass claim")
    cases = operators["cases"]
    require(len(cases) == 73 and len({c["case"] for c in cases}) == 73,
            "Expected 73 distinct operator checks")
    exact = rejected = 0
    for case in cases:
        require(("exact" in case) != ("rejected" in case),
                "Ambiguous operator result")
        if "exact" in case:
            require(case["exact"] is True and type(case["elements"]) is int
                    and case["elements"] >= 0, "Operator numerical check did not pass")
            exact += 1
        else:
            require(case["rejected"] is True, "Invalid input was not rejected")
            rejected += 1
    require((exact, rejected) == (66, 7)
            and operators["case_count"] == 73
            and operators["exact_numeric_case_count"] == exact
            and operators["invalid_input_rejection_case_count"] == rejected,
            "Operator count mismatch")
    positive(operators["seconds"], "operator duration")


def check_original_hashes(data):
    """Reconstruct the two retained source reports, with their original JSON endings."""
    provenance = data["provenance"]
    original = {key: data[key] for key in ORIGINAL_SUMMARY_KEYS}
    operators = {key: value for key, value in data["operator_checks"].items()
                 if key not in {"case_count", "exact_numeric_case_count",
                                "invalid_input_rejection_case_count"}}
    for value, ending, key in (
        (original, "", "original_model_summary_SHA256"),
        (operators, "\n", "original_GPU_operator_report_SHA256"),
    ):
        raw = (json.dumps(value, ensure_ascii=False, indent=2) + ending).encode("utf-8")
        require(hashlib.sha256(raw).hexdigest() == provenance[key],
                "Retained original report SHA256 mismatch: " + key)
    require(is_sha(provenance["original_benchmark_driver_SHA256"]),
            "Invalid original driver SHA256")


def validate_record(data, root):
    require(data["format"] == "q38-inference-candidate-matched-v1"
            and data["updates"] == 96 and data["packed_manifest_SHA256"] == PACKED_SHA256
            and data["prompt_SHA256"] == PROMPTS_SHA256, "Wrong recorded artifact")
    require(data["passed"] is True and data["quality_accepted"] is False
            and data["weights_changed"] is False, "Unsupported overall pass claim")
    require(data["release_context"]["public_step32_fast_path_GPU_verified_by_this_record"] is False
            and data["release_context"]["direct_packed_GEMV_tested"] is False,
            "Unsupported release or implementation claim")
    check_sources(root, data)
    prompts = read_json(Path(root) / COURSE, "diagnostic-prompts.json")
    require(len(prompts) == 8 and len({p["id"] for p in prompts}) == 8
            and all(p["max_new_tokens"] == 32 for p in prompts), "Wrong prompt set")
    ids = [p["id"] for p in prompts]
    require(data["round_order"] == ORDER and len(data["rounds"]) == 4
            and len(data["derived_summary"]) == 4, "Expected four ordered rounds")
    details = data["measurement_details"]
    require(details["batch_size"] == 1 and details["development_prompts"] == 8
            and details["maximum_input_tokens_including_chat_template"] == 128
            and details["maximum_new_tokens"] == 32 and details["greedy"] is True
            and details["dynamic_KV_cache"] is True and details["enable_thinking"] is False,
            "Wrong measurement conditions")
    reference = data["rounds"][0]["cases"]
    for index, (round_data, mode, derived) in enumerate(
            zip(data["rounds"], ORDER, data["derived_summary"])):
        cases = round_data["cases"]
        require(round_data["mode"] == mode and len(cases) == 8
                and [c["id"] for c in cases] == ids, "Round/case order mismatch")
        for i, case in enumerate(cases):
            label = "round {} {}".format(index, case["id"])
            require(case["prompt"] == prompts[i]["prompt"], "Prompt mismatch: " + label)
            tokens(case["input_token_IDs"], "input tokens", 128)
            tokens(case["generated_token_IDs"], "generated tokens", 32)
            times, hashes = case["step_seconds"], case["last_position_logit_SHA256"]
            require(len(times) == len(hashes) == len(case["generated_token_IDs"]),
                    "Token/timing/logit count mismatch: " + label)
            require(all(is_sha(h) for h in hashes), "Invalid logits SHA256: " + label)
            require(type(case["ended_with_EOS"]) is bool
                    and isinstance(case["text"], str), "Malformed generation: " + label)
            for elapsed in times:
                positive(elapsed, "token duration")
            close(case["first_token_seconds"], times[0], "first token: " + label)
            if len(times) > 1:
                close(case["decode_after_first_tokens_per_second"],
                      (len(times) - 1) / sum(times[1:]), "case decode rate: " + label)
            else:
                require(case["decode_after_first_tokens_per_second"] is None,
                        "A single token has no post-first rate")
            allocated, reserved = case["peak_allocated_bytes"], case["peak_reserved_bytes"]
            require(type(allocated) is int and type(reserved) is int
                    and 0 < allocated <= reserved, "Invalid memory counters: " + label)
            if index:
                for key in ("input_token_IDs", "generated_token_IDs", "text",
                            "ended_with_EOS", "last_position_logit_SHA256"):
                    require(case[key] == reference[i][key],
                            "Reference mismatch: " + label + " " + key)
                require(case["reference_token_IDs_exact"] is True
                        and case["reference_all_step_logits_bytes_exact"] is True,
                        "Missing exact-match claim: " + label)
        count = sum(len(c["generated_token_IDs"]) - 1 for c in cases)
        total = sum(sum(c["step_seconds"][1:]) for c in cases)
        generated = sum(len(c["generated_token_IDs"]) for c in cases)
        require(count == details["timed_decode_tokens_per_round"] == 87
                and generated == details["generated_tokens_including_first_and_EOS_per_round"] == 95,
                "Pooled token count mismatch")
        close(round_data["pooled_decode_tokens_per_second"], count / total,
              "pooled decode rate")
        require(derived["round"] == index and derived["mode"] == mode
                and derived["prompts_completed"] == 8 and derived["post_first_tokens"] == count
                and derived["generated_tokens_and_per_step_last_position_logits_match_first_reference"] is True,
                "Derived summary identity or pass mismatch")
        allocated = max(c["peak_allocated_bytes"] for c in cases)
        reserved = max(c["peak_reserved_bytes"] for c in cases)
        for name, expected in {
            "post_first_compute_seconds": total,
            "pooled_decode_tokens_per_second": count / total,
            "median_first_token_compute_seconds": statistics.median(c["step_seconds"][0] for c in cases),
            "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
            "peak_allocated_GiB": allocated / 2**30, "peak_reserved_GiB": reserved / 2**30,
        }.items():
            close(derived[name], expected, "derived " + name)
    semantics = data["semantic_review"]
    require(semantics["quality_accepted"] is False
            and [c["id"] for c in semantics["cases"]] == ids,
            "Unsupported semantic acceptance claim")
    check_operators(data)
    check_original_hashes(data)
    check_public_provenance(root, data)
    return "4 x 8 matched cases; 87 timed decode tokens/round; 73 operator checks (66 + 7)"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    try:
        result = validate_record(read_json(args.root, EVIDENCE), args.root)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        print("FAIL: " + str(exc), file=sys.stderr)
        return 1
    print("PASS: " + result)
    print("CPU consistency checks only; do not rerun GPU or establish model quality.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
