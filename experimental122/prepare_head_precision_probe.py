"""Copy the GPU-tested FP32 head diagnostic into a new isolated package.

Standard library only. Run the copied HTTP entry with BONSAI_HEAD_FP32=1.
This prepares source; it does not start a GPU engine or verify new requests.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    provenance = package / 'vllm-http-provenance.json'
    manifest = json.loads(provenance.read_text(encoding='utf-8'))
    policy = manifest['file_sha256']
    assert len(policy) == 68 and all(sha(package.parent / n) == h for n, h in policy.items())
    helper = package / 'runtime/head_fp32_bound_candidate_v1871.py'
    helper_sha = '2e35944de371796de9996227681cf77faffbb0626664182d14392f579f00ca34'
    assert sha(helper) == helper_sha
    destination = args.output.resolve()
    assert not destination.exists()
    destination.mkdir(parents=True)
    names = [*policy, 'experimental122/vllm-prefill-block16-provenance.json',
             'experimental122/vllm-b4-window-provenance.json']
    assert sha(package / 'vllm-prefill-block16-provenance.json') == '4ff8e3546882d527e0f2ecfaad091f426e26b751f0ca4b2a99909ecc5e8ae380'
    for name in names:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((package.parent / name).read_bytes())
    target_helper = destination / 'experimental122/runtime' / helper.name
    target_helper.write_bytes(helper.read_bytes())
    plugin = destination / 'experimental122/http_plugin/bonsai_http_plugin_v1.py'
    source = plugin.read_text(encoding='utf-8')
    source += """
    if os.environ.get('BONSAI_HEAD_FP32')=='1':
        candidate=package/'runtime/head_fp32_bound_candidate_v1871.py'
        assert sha(candidate)==CANDIDATE_SHA
        from head_fp32_bound_candidate_v1871 import install
        install()
        INFO['head_fp32_candidate_installed']=True
""".replace('CANDIDATE_SHA', repr(helper_sha))
    plugin.write_text(source, encoding='utf-8', newline='\n')
    manifest['file_sha256']['experimental122/http_plugin/bonsai_http_plugin_v1.py'] = sha(plugin)
    manifest['scope'] = 'Private instance-bound FP32 head candidate; original engine64 and public entry unchanged. Actual GPU call dtype proof required.'
    target = destination / 'experimental122/vllm-http-provenance.json'
    target.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    for path in (plugin, target_helper):
        ast.parse(path.read_text(encoding='utf-8'))
    print(json.dumps({'prepared': True, 'copied_HTTP_source_files': 68,
                      'head_helper_SHA256': sha(target_helper),
                      'candidate_plugin_SHA256': sha(plugin),
                      'weights_modified': False, 'default_entry_modified': False,
                      'GPU_engine_started': False, 'requires_BONSAI_HEAD_FP32_1': True}))


if __name__ == '__main__':
    main()
