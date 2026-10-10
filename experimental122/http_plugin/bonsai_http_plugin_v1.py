"""Process-local registration in the API server and its spawned engine workers."""
import hashlib
import json
import os
from pathlib import Path
import sys

INFO = {}


def register():
    if INFO:
        return
    root = Path(__file__).resolve().parents[2]
    package = root / 'experimental122'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = package / 'vllm-prefill-block16-provenance.json'
    assert sha(manifest) == '4ff8e3546882d527e0f2ecfaad091f426e26b751f0ca4b2a99909ecc5e8ae380'
    engine = json.loads(manifest.read_text(encoding='utf-8'))['file_sha256']
    policy = json.loads((package / 'vllm-http-provenance.json').read_text(encoding='utf-8'))['file_sha256']
    assert len(engine) == 64 and len(policy) == 68
    assert all(sha(root / n) == h for n, h in engine.items())
    assert all(sha(root / n) == h for n, h in policy.items())
    sys.path.insert(0, str(package / 'runtime'))
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    from bonsai_native122_vllm_b4_window_v1689 import register_model, ARCHITECTURE, LOAD_FORMAT
    from bonsai_native122_vllm_dense_v1576 import NAME
    register_model()
    grouped = os.environ.get('BONSAI_HTTP_GROUPED_PREFILL') == '1'
    if grouped:
        from bonsai_vllm_prefill_block16_candidate_v1747 import install
        install()
    INFO.update(registered=True, architecture=ARCHITECTURE, load_format=LOAD_FORMAT,
                quantization=NAME, engine64_and_HTTP68_source_files_verified=True,
                grouped_prefill_installed=grouped,
                multiprocessing_environment=os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING'))
