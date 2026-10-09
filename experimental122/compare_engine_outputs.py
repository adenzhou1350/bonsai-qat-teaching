"""Compare saved native/vLLM token outputs on CPU; no timing or quality claim.

Model buffers and cache layouts may differ between engines. This checks result
metadata and emitted IDs, not the loaded tensors or the original prompt text.
"""
import argparse
import json
import re
from pathlib import Path


def compare(baseline, candidate):
    for data in (baseline, candidate):
        assert data['completed'] and data['eos_respected'] and not data['CPU_weight_offload']
        rows = data['requests']
        assert rows and [r['request_index'] for r in rows] == list(range(len(rows)))
        assert re.fullmatch(r'[0-9a-f]{64}', data['artifact_manifest_sha256'])
        for row in rows:
            ids = row['generated_token_ids']
            assert ids and all(type(x) is int and x >= 0 for x in ids)
            assert row['output_tokens'] == len(ids) <= data['max_new_tokens']
            assert type(row['input_tokens']) is int and row['input_tokens'] > 0
            assert type(row['eos_reached']) is bool
    for key in ('artifact_manifest_sha256', 'max_new_tokens'):
        assert baseline[key] == candidate[key], ('Different metadata', key)
    assert len(baseline['requests']) == len(candidate['requests']), 'Different request counts'
    for original, new in zip(baseline['requests'], candidate['requests']):
        for key in ('input_tokens', 'generated_token_ids', 'eos_reached'):
            assert original[key] == new[key], ('Request differs', original['request_index'], key)
    return {
        'all_saved_output_token_IDs_equal': True,
        'input_token_counts_and_EOS_match': True,
        'artifact_manifest_metadata_matches': True,
        'requests': len(baseline['requests']),
        'output_tokens_per_result': sum(r['output_tokens'] for r in baseline['requests']),
        'limits': 'Saved JSON comparison only. Does not load weights, verify prompt text, measure throughput/memory, or establish model quality.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    args = parser.parse_args()
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))
    print(json.dumps(compare(load(args.baseline), load(args.candidate)), indent=2))


if __name__ == '__main__':
    main()
