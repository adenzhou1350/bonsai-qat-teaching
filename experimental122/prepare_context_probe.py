"""Prepare an isolated copy of the tested 1408-token context diagnostic.

CPU only, standard library only. Leaves the default short-input entry unchanged.
The copied engine still has maxlen1536, maxseq4, cap<=64 and cache>=896MiB.
Four retrieval probes establish only their measured scope, not general quality.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New directory for the isolated diagnostic package.')
    args=parser.parse_args()
    package=Path(__file__).resolve().parent
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=package/'vllm-prefill-block16-provenance.json'
    assert sha(manifest)=='4ff8e3546882d527e0f2ecfaad091f426e26b751f0ca4b2a99909ecc5e8ae380'
    policy=json.loads(manifest.read_text(encoding='utf-8'))['file_sha256']
    assert len(policy)==64 and all(sha(package.parent/n)==h for n,h in policy.items())
    destination=args.output.resolve()
    assert not destination.exists()
    destination.mkdir(parents=True)
    names=[*policy,'experimental122/vllm-b4-window-provenance.json','experimental122/vllm-prefill-block16-provenance.json']
    for n in names:
        p=destination/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((package.parent/n).read_bytes())
    base=destination/'experimental122/run_vllm_requests_b4_window.py'
    source=base.read_text(encoding='utf-8')
    assert source.count('assert max(map(len, inputs)) <= 128')==1
    source=source.replace('assert max(map(len, inputs)) <= 128','assert max(map(len, inputs)) <= 1408').replace('maximum128 input tokens','private diagnostic maximum1408 input tokens').replace('One queued generate call; short prompts','Private chunked-context boundary probe. One queued generate call; longer prompts')
    base.write_text(source,encoding='utf-8',newline='\n')
    wrapper=destination/'experimental122/run_vllm_requests_b4_prefill.py'
    wrapper.write_text(wrapper.read_text(encoding='utf-8').replace('Maximum128 input tokens','Private diagnostic maximum1408 input tokens'),encoding='utf-8',newline='\n')
    for name,count in [('vllm-b4-window-provenance.json',62),('vllm-prefill-block16-provenance.json',64)]:
        path=destination/'experimental122'/name
        data=json.loads(path.read_text(encoding='utf-8'));mapping=data['file_sha256'];assert len(mapping)==count
        for n in mapping:mapping[n]=sha(destination/n)
        data['private_diagnostic']='Only input guard/description widened; not validated or published as long-input support.'
        path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n')
    for n in policy:
        if n.endswith('.py'):ast.parse((destination/n).read_text(encoding='utf-8'))
    assert sha(base)=='6e84e8797f1c657203610817a6e23173e4fb1f5e3336d4a5fcd0392d3aacd99a'
    print(json.dumps({'prepared':True,'source_files':64,'maximum_input_tokens':1408,'max_model_len':1536,'weights_modified':False,'default_entry_modified':False,'engine_run_by_this_helper':False}))


if __name__=='__main__':
    main()
