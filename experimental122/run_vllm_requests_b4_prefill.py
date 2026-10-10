"""Experimental grouped-prefill16 wrapper for the validated B4 window entry.

Same CLI arguments and cache896 default as run_vllm_requests_b4_window.py.
Grouped GEMM changes prefill rounding; short-fixture equivalence is not a broad
quality or long-input guarantee. Decode uses the existing small-batch path.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    package = Path(__file__).resolve().parent
    sys.path.insert(0, str(package))
    import run_vllm_requests_b4_window as core
    if '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
        core.main()
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    provenance = package / 'vllm-prefill-block16-provenance.json'
    policy = json.loads(provenance.read_text(encoding='utf-8'))['file_sha256']
    assert len(policy) == 64
    assert all(sha(package.parent / name) == digest for name, digest in policy.items())
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['BONSAI_BATCHED_PREFILL'] = '1'
    sys.path.insert(0, str(package / 'runtime'))
    from bonsai_vllm_prefill_block16_candidate_v1747 import install, restore, AUDIT
    info, originals = install()
    try:
        core.main()
        output = args.output.resolve()
        path = output / 'result.json'
        report = json.loads(path.read_text(encoding='utf-8'))
        assert report['completed'] and AUDIT['gate_prefill_calls'] > 0
        assert AUDIT['maximum_decoded_expert_chunk'] == 16
        assert all(sha(package.parent / name) == digest for name, digest in policy.items())
        report.update(grouped_prefill_wrapper_completed=True,
                      approximate_prefill_provider=info,
                      approximate_prefill_audit=dict(AUDIT),
                      grouped_prefill_source_policy_sha256=sha(provenance),
                      grouped_prefill_all64_source_files_verified=True,
                      quality_accepted=False,
                      limits='Experimental grouped-prefill16 with unchanged packed weights and small-batch decode; grouped BF16 GEMM changes prefill rounding. Maximum128 input tokens, cap64, maxseq4/maxlen1536. Generate timing excludes model loading and wrapper JSON processing. Short-fixture equivalence does not establish independent quality, HTTP, long-input or sustained-service acceptance.')
        (output / 'grouped-prefill-source-policy.json').write_text(json.dumps(policy, indent=2))
        temporary = output / 'result.tmp'
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        temporary.replace(path)
        print(json.dumps({'grouped_prefill_wrapper_completed': True,
                          'changes_prefill_arithmetic': True,
                          'all64_source_files_verified': True,
                          'quality_accepted': False}))
    finally:
        restore(originals)


if __name__ == '__main__':
    main()
