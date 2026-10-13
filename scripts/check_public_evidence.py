"""Check published file identities and summary arithmetic, without running a model."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(root, name):
    return json.loads((root / name).read_text(encoding="utf-8"))


def check_hash(root, name, expected):
    path = (root / name).resolve()
    require(path.is_relative_to(root), f"Path outside checkout: {name}")
    require(path.is_file(), f"Missing public file: {name}")
    actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    require(actual == expected, f"SHA256 mismatch: {name}")


def close(actual, expected, label):
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10),
            f"Summary arithmetic mismatch: {label}")


def check_http(data):
    runs = data["runs"]
    require([r["mode"] for r in runs] == ["original", "grouped", "grouped", "original"],
            "Expected the recorded four-round HTTP ABBA protocol")
    rates = {}
    for mode in ("original", "grouped"):
        selected = [r for r in runs if r["mode"] == mode]
        require(all(r["client_wall_seconds"] > 0 for r in selected), "Nonpositive HTTP time")
        rates[mode] = sum(r["output_tokens"] for r in selected) / sum(
            r["client_wall_seconds"] for r in selected)
        close(rates[mode], data["pooled_measured_output_tokens_per_second"][mode], mode)
    change = (rates["grouped"] / rates["original"] - 1) * 100
    close(change, data["observed_change_percent"], "HTTP relative change")
    requests = sum(r["measured_requests"] for r in runs)
    exact = sum(r["old_CLI_exact_ID_count"] for r in runs)
    require(all(0 <= r["old_CLI_exact_ID_count"] <= r["measured_requests"] for r in runs),
            "Invalid HTTP request counts")
    require(exact == data["measured_requests_exact_to_old_CLI"], "HTTP exact-ID count mismatch")
    require(requests - exact == len(data["different_wording_requests"]),
            "HTTP differing-request count mismatch")
    require(sum(r["output_tokens"] for r in runs) == data["measured_output_IDs"],
            "HTTP output-token count mismatch")
    profiles = [r["NVML_v2_profile"] for r in runs]
    require(sum(p["samples"] for p in profiles) == data["NVML_rows_recounted"],
            "NVML sample count mismatch")
    used = max(p["maximum_observed_used_bytes"] for p in profiles) / 2**30
    free = min(p["minimum_observed_free_bytes"] for p in profiles) / 2**20
    close(used, data["maximum_observed_NVML_used_GiB"], "NVML used GiB")
    close(free, data["minimum_observed_NVML_free_MiB"], "NVML free MiB")
    for p in profiles:
        for key in ("driver_reserved_bytes_min", "driver_reserved_bytes_max"):
            close(p[key] / 2**30, data["driver_reserved_GiB_separate"], "driver reserved GiB")
    require(data["exact_ID_acceptance_claimed"] is False and data["quality_accepted"] is False,
            "This snapshot does not establish exact-ID or quality acceptance")
    return [
        f"HTTP pooled output: {rates['original']:.2f} -> {rates['grouped']:.2f} tokens/s "
        f"({change:+.2f}%; aggregate of four clients)",
        f"HTTP complete requests matching old CLI: {exact}/{requests}; "
        f"{requests - exact} differ (not a token accuracy score)",
        f"HTTP sampled NVML used: {used:.2f} GiB; minimum free: {free:.2f} MiB; "
        f"separate driver reserved: {data['driver_reserved_GiB_separate']:.3f} GiB",
    ]


def check_quality(r35, r122):
    expected_denominators = {"human": 164, "cn": 48, "cmmlu": 201, "gsm8k": 200}
    lines = []
    panels122 = r122["public_development_evaluation"]["panels"]
    for label, panels in (("R35-8192", r35["completed_public_panels"]),
                          ("R122-native4095", panels122)):
        require(len(panels) == 4 and {p["mode"] for p in panels} == set(expected_denominators),
                f"{label}: expected four recorded development panels")
        all_noninferior = True
        values = []
        for p in panels:
            if label == "R35-8192":
                n, candidate, baseline = p["all_cases"], p["passes"], p["native_BF16_passes"]
            else:
                n, candidate, baseline = p["completed"], p["new_candidate_passes"], p["native_original_passes"]
                require(sum(p[k] for k in ("both_passed", "source_passes_lost", "new_passes", "both_failed")) == n,
                        f"{label}/{p['mode']}: paired counts do not sum to denominator")
                require(candidate == p["both_passed"] + p["new_passes"] and
                        baseline == p["both_passed"] + p["source_passes_lost"],
                        f"{label}/{p['mode']}: paired pass counts disagree")
                require(p["aggregate_noninferior"] == (candidate >= baseline),
                        f"{label}/{p['mode']}: noninferiority flag disagrees")
            require(n == expected_denominators[p["mode"]] and 0 <= candidate <= n and 0 <= baseline <= n,
                    f"{label}/{p['mode']}: invalid counts")
            all_noninferior &= candidate >= baseline
            values.append(f"{p['mode']} {candidate}/{n} vs {baseline}/{n}")
        require(not all_noninferior, f"{label}: unexpected all-panel noninferiority")
        lines.append(f"{label} development panels (candidate vs matched BF16): " + "; ".join(values))
    require(r35["quality_accepted"] is False and r122["quality_accepted"] is False,
            "Quality acceptance flags differ from this snapshot")
    require(r122["public_development_evaluation"]["all_four_aggregate_noninferior"] is False,
            "R122 all-panel flag disagrees")
    return lines


def check(root):
    root = root.resolve()
    manifest = read_json(root, "docs/public-evidence-manifest.json")
    require(manifest["format"] == "bonsai-public-evidence-v1", "Unsupported evidence manifest")
    for name, expected in manifest["file_sha256"].items():
        check_hash(root, name, expected)
    sources = {}
    for entry in manifest["source_manifests"]:
        for name, expected in read_json(root, entry["path"])["file_sha256"].items():
            relative = (Path(entry["file_prefix"]) / name).as_posix()
            require(relative not in sources or sources[relative] == expected,
                    f"Conflicting source identities: {relative}")
            check_hash(root, relative, expected)
            sources[relative] = expected
    http = read_json(root, manifest["records"]["http"])
    r35 = read_json(root, manifest["records"]["r35_quality"])
    r122 = read_json(root, manifest["records"]["r122_training_quality"])
    artifact = r122["artifact_manifest_sha256"]
    require(http["artifact_manifest_SHA256"] == artifact, "HTTP/training artifact identities differ")
    require(read_json(root, "experimental122/vllm-http-provenance.json")["artifact_manifest_sha256"] == artifact,
            "HTTP source manifest targets a different artifact")
    lines = check_http(http) + check_quality(r35, r122)
    return [
        "PASS: public file identities and summary arithmetic only",
        f"Evidence/source baseline commit: {manifest['baseline_commit']}",
        f"Pinned public files: {len(manifest['file_sha256'])}; source files: {len(sources)} unique",
        *lines,
        "No model, GPU, raw request logs, evaluation samples, or weights were executed/verified.",
        "This check does not reproduce training, quality scores, or GPU performance.",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Checkout to inspect (default: the checkout containing this script)")
    args = parser.parse_args()
    try:
        print("\n".join(check(args.root)))
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        print(f"FAIL: public evidence check: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
