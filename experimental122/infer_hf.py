"""Run the public native4095 release with its sanitized metadata identity.

Only the expected manifest identity changes. The packed tensor files, format,
hash verification and numerical runtime are the previously recorded artifact.
"""
import argparse
import json
import sys

import infer

RELEASE_MANIFEST_SHA256 = 'da42f0bf48b42f2885061283d83f9da72f9429ad8ed74a9691e8cc532f15a3ef'
ORIGINAL_MANIFEST_SHA256 = 'dc3a9ba1a8bd9bcdd229b3c259cc5d676719861946a5ed2d90b39ac2e196def1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--verify-only', action='store_true', help='Verify the manifest and all tensor files without CUDA')
    parser.add_argument('--prompt', default='用中文解释什么是模型量化。')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--contexts', type=int, nargs='+', default=[128, 512, 4096, 8192, 16384])
    parser.add_argument('--output')
    args = parser.parse_args()
    expected, format_tag = infer.ARTIFACTS['native4095']
    if expected != ORIGINAL_MANIFEST_SHA256:
        raise ValueError('The underlying native4095 entry has changed')
    infer.ARTIFACTS['native4095'] = (RELEASE_MANIFEST_SHA256, format_tag)
    if args.verify_only:
        _, report = infer.manifest(args.model, 'native4095')
        print(json.dumps({
            'release_manifest_and_all_tensor_file_SHA256_verified': True,
            'tensor_files': len(report['matrices']) + 2,
            'cumulative_optimizer_updates': report['actual_optimizer_updates'],
            'GPU_used': False,
            'new_GPU_inference_or_quality_acceptance_claimed': False,
        }))
        return
    sys.argv.extend(['--artifact-version', 'native4095'])
    infer.main()


if __name__ == '__main__':
    main()
