"""Single-request / terminal-chat R35-8192 release inference with unchanged kernels."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

RELEASE_MANIFEST = '48e3f2479ef174a93c043d900bc163e127969e9ee522bf8c084cd080ae26dd16'
RELEASE_FILES = 'de459963814ec3263b7baaffa5a5059b5602911652886d225699dd6d870309bb'
FORMAT = 'research-gsq35-int2-rank16-native4095-adam-second-epoch8192-v1'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def verify(directory):
    directory = Path(directory).resolve()
    if sha(directory / 'summary.json') != RELEASE_MANIFEST:
        raise ValueError('Expected the sanitized R35-8192 release, not T35 or another model')
    report = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    if report['format'] != FORMAT or report['actual_full_gpu_optimizer_updates'] != 8192 or not report['passed']:
        raise ValueError('Incomplete or incompatible artifact')
    if sha(directory.parent / 'release-files.json') != RELEASE_FILES:
        raise ValueError('The release payload inventory has changed')
    inventory = json.loads((directory.parent / 'release-files.json').read_text(encoding='utf-8'))
    rows = inventory['files']
    if inventory['tensor_files'] != 81 or sum(n.endswith('.pt') for n in rows) != 81:
        raise ValueError('Expected all80 expert banks and retained tensors')
    for name, row in rows.items():
        path = (directory.parent / name).resolve()
        if not path.is_relative_to(directory.parent) or path.stat().st_size != row['bytes'] or sha(path) != row['SHA256']:
            raise ValueError('Release file identity mismatch: ' + name)
    expected = {x['file']: x['sha256'] for x in report['matrices']}
    if len(expected) != 80 or any(rows['packed/' + n]['SHA256'] != h for n, h in expected.items()):
        raise ValueError('Expert manifest differs from release inventory')
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True, help='Downloaded release packed/ directory')
    p.add_argument('--verify-only', action='store_true', help='Hash the release without importing Torch or CUDA')
    p.add_argument('--prompt', default='用中文解释什么是模型量化。')
    p.add_argument('--chat', action='store_true', help='One terminal chat; /reset clears history, /exit quits')
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--context', type=int, default=4096, help='Input plus output budget, default4096, maximum16384')
    p.add_argument('--benchmark', action='store_true', help='Fixed64 decode steps; continues beyond EOS')
    p.add_argument('--contexts', type=int, nargs='+', default=[128, 512, 4096, 8192, 16384])
    p.add_argument('--output', type=Path, help='New JSON output path; never overwrite an earlier result')
    return p


def main():
    a = parser().parse_args()
    if not 1 <= a.max_new_tokens <= 1024 or not 128 <= a.context <= 16384:
        raise ValueError('Use1..1024 new tokens and128..16384 total context')
    if a.chat and a.benchmark or any(not 1 <= n <= 16384 for n in a.contexts):
        raise ValueError('Choose chat or benchmark and supported contexts')
    if a.output and a.output.exists():
        raise FileExistsError('Choose a new output file')
    report = verify(a.model)
    if a.verify_only:
        print(json.dumps({'release_manifest_and_all_tensor_file_SHA256_verified': True,
                          'tensor_files': 81, 'cumulative_optimizer_updates': 8192, 'GPU_used': False,
                          'new_GPU_inference_or_quality_acceptance_claimed': False}))
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'runtime'))
    import torch
    from portable_loader import load_packed
    from gsq_graph_decoder_v2 import GraphDecoder
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen
    if torch.cuda.device_count() != 1 or '5090' not in torch.cuda.get_device_name(0):
        raise ValueError('Select one idle32GB RTX5090 with CUDA_VISIBLE_DEVICES')
    if qwen.causal_conv1d_update is None:
        raise RuntimeError('The recorded decode path requires causal-conv1d1.7.0 CUDA extension; see deployment guide')
    torch.set_num_threads(1)
    tok, model, _ = load_packed(a.model, report)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) | {tok.eos_token_id}
    result = {'format': 'R35-release-inference-v1', 'optimizer_updates': 8192,
              'manifest_SHA256': RELEASE_MANIFEST, 'quality_accepted': False, 'requests': [], 'benchmarks': []}
    with torch.inference_mode():
        if a.benchmark:
            import gc
            pattern = tok.encode('This is a controlled timing context for a language model. ', add_special_tokens=False)
            for length in a.contexts:
                ids = torch.tensor((pattern * ((length + len(pattern) - 1) // len(pattern)))[:length], device='cuda')[None]
                torch.cuda.reset_peak_memory_stats()
                decoder = GraphDecoder(model, ids, max_cache_len=length + 128)
                decoder.prepare(ids)
                torch.cuda.synchronize(); start = time.perf_counter()
                generated = []
                for _ in range(64):
                    decoder.graph.replay(); generated.append(decoder.token.clone())
                torch.cuda.synchronize(); seconds = time.perf_counter() - start
                actual = torch.cat(generated, -1)
                decoder.prepare(ids); static = []
                for _ in range(64):
                    decoder.step(); static.append(decoder.token.clone())
                if not torch.equal(actual, torch.cat(static, -1)):
                    raise ValueError('Graph/static token mismatch')
                row = {'context_tokens': length, 'fixed_decode_steps': 64, 'decode_tokens_per_second': 64 / seconds,
                       'graph_static_equal': True, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                       'timing_excludes_load_prefill_capture': True, 'continues_past_EOS': True}
                result['benchmarks'].append(row); print(json.dumps(row), flush=True)
                del decoder, ids, generated, actual, static
                gc.collect(); torch.cuda.empty_cache()
        else:
            history = []
            decoder = None
            while True:
                if a.chat:
                    try:
                        prompt = input('You> ').strip()
                    except (EOFError, KeyboardInterrupt):
                        break
                    if prompt == '/exit': break
                    if prompt == '/reset': history = []; continue
                    if not prompt: continue
                else:
                    prompt = a.prompt
                messages = history + [{'role': 'user', 'content': prompt}]
                text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                tokens = tok.encode(text, add_special_tokens=False)
                if len(tokens) + max(a.max_new_tokens, 5) > a.context:
                    if a.chat:
                        print('Context budget exceeded. Use /reset or restart with a larger --context.'); continue
                    raise ValueError('Input plus output exceeds --context; use a shorter prompt or a larger context')
                ids = torch.tensor(tokens, device='cuda')[None]
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize(); started = time.perf_counter()
                if decoder is None:
                    decoder = GraphDecoder(model, ids, max_cache_len=a.context + 128)
                generated = decoder.generate_ids(ids, a.max_new_tokens, eos)
                torch.cuda.synchronize(); elapsed = time.perf_counter() - started
                answer = tok.decode(generated, skip_special_tokens=True)
                print(('Assistant> ' if a.chat else '') + answer, flush=True)
                result['requests'].append({'input_tokens': len(tokens), 'generated_token_ids': generated, 'text': answer,
                     'ended_with_EOS': bool(generated and generated[-1] in eos),
                     'request_seconds_includes_prepare': elapsed, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
                history = messages + [{'role': 'assistant', 'content': answer}]
                if not a.chat: break
    result['completed'] = True
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
