"""Experimental single-5090 OpenAI server for the exact native4095 packed model.

Use --model PACKED --output NEW_DIRECTORY [--grouped-prefill]. Other arguments
are forwarded to the installed vLLM 0.24 serve parser; validated limits remain
maxlen1536/maxseq4/cache896MiB. No model weights are included in this repository.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--grouped-prefill', action='store_true')
    options, remaining = parser.parse_known_args()
    package = Path(__file__).resolve().parent
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    provenance = package / 'vllm-http-provenance.json'
    policy = json.loads(provenance.read_text(encoding='utf-8'))['file_sha256']
    assert len(policy) == 68 and all(sha(package.parent / n) == h for n, h in policy.items())
    artifact = options.model.resolve()
    assert sha(artifact / 'summary.json') == 'dc3a9ba1a8bd9bcdd229b3c259cc5d676719861946a5ed2d90b39ac2e196def1'
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'HTTP-source-policy.json').write_text(json.dumps(policy, indent=2))
    plugin = package / 'http_plugin'
    sys.path.insert(0, str(plugin))
    os.environ['PYTHONPATH'] = str(plugin) + (os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')
    os.environ.update(VLLM_PLUGINS='bonsai_native122_http',
                      VLLM_ENABLE_V1_MULTIPROCESSING='1',
                      VLLM_WORKER_MULTIPROC_METHOD='spawn',
                      BONSAI_BATCHED_PREFILL='1',
                      BONSAI_HTTP_GROUPED_PREFILL='1' if options.grouped_prefill else '0')
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    import bonsai_http_plugin_v1 as registered
    from vllm.entrypoints.openai.api_server import run_server
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from bonsai_native122_vllm_b4_window_v1689 import text_window_overrides
    defaults = ['--model', str(artifact), '--host', '127.0.0.1', '--port', '8000',
                '--served-model-name', 'bonsai-native122', '--dtype', 'bfloat16',
                '--quantization', registered.INFO['quantization'],
                '--load-format', registered.INFO['load_format'],
                '--model-loader-extra-config', json.dumps({'weight_proof': str(output / 'weight-proof.json')}),
                '--max-model-len', '1536', '--max-num-seqs', '4',
                '--max-num-batched-tokens', '256', '--kv-cache-memory-bytes', str(896 * 1024**2),
                '--distributed-executor-backend', 'uni', '--no-enable-prefix-caching',
                '--generation-config', 'vllm', '--seed', '1614', '--shutdown-timeout', '30',
                '--default-chat-template-kwargs', json.dumps({'enable_thinking': False}),
                '--compilation-config', json.dumps({'mode': 0, 'cudagraph_mode': 'FULL_DECODE_ONLY',
                                                    'cudagraph_capture_sizes': [1, 2, 4]})]
    args = make_arg_parser(FlexibleArgumentParser()).parse_args(defaults + remaining)
    validate_parsed_serve_args(args)
    assert args.host == '127.0.0.1' and args.max_model_len == 1536 and args.max_num_seqs == 4
    assert args.max_num_batched_tokens == 256 and args.kv_cache_memory_bytes == 896 * 1024**2
    assert args.quantization == registered.INFO['quantization'] and args.load_format == registered.INFO['load_format']
    assert args.shutdown_timeout > 0 and args.cpu_offload_gb == 0 and args.tensor_parallel_size == 1
    args.hf_overrides = lambda config: text_window_overrides(config, artifact)
    asyncio.run(run_server(args))


if __name__ == '__main__':
    main()
