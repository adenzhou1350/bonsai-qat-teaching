"""Arbitrary short text prompt for the recorded Q38 packed correctness runtime."""
import argparse
import hashlib
import json
import time
from pathlib import Path

KNOWN = {'7217d5c218e4b27317e9862224113d8bf694944329d8a70c80c9ae53742a03d4': 32,
         'c53fa86911cd61f85321d1305fcab03c39a45c38fa5f2fc18d4cdb982988eed9': 96}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(8 * 1024 * 1024): h.update(b)
    return h.hexdigest()


def verify(root):
    root = root.resolve(); identity = sha(root / 'manifest.json')
    if identity not in KNOWN: raise ValueError('Expected a recorded Q38 step32 or step96 packed manifest')
    v = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if v['actual_optimizer_updates'] != KNOWN[identity] or len(v['matrices']) != 402:
        raise ValueError('Unexpected update count or quantization scope')
    files = {r['file']: r['SHA256'] for r in v['matrices']}
    files.update({'config.json': v['config_SHA256'], 'retained.pt': v['retained_SHA256']})
    for name, digest in files.items():
        p = (root / name).resolve()
        if not p.is_relative_to(root) or sha(p) != digest: raise ValueError('Payload hash mismatch: ' + name)
    return v, identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packed', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path)
    p.add_argument('--prompt', default='用一句话解释模型量化。')
    p.add_argument('--max-new-tokens', type=int, default=32)
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if not 1 <= a.max_new_tokens <= 64: raise ValueError('Use1..64 new tokens for this short reference path')
    if a.output and a.output.exists(): raise FileExistsError('Choose a new output file')
    report, identity = verify(a.packed)
    if a.verify_only:
        print(json.dumps({'payload_verified': True, 'matrix_files': 402, 'optimizer_updates': report['actual_optimizer_updates'], 'GPU_used': False})); return
    if a.tokenizer is None: raise ValueError('--tokenizer is required for generation')
    import torch
    from transformers import AutoTokenizer
    from control import packed_reload
    if torch.cuda.device_count() != 1: raise ValueError('Expose one admitted CUDA GPU')
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': a.prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if len(tokens) > 128: raise ValueError('This reference entry supports at most128 input tokens including the chat template')
    model, _ = packed_reload(a.packed, torch.device('cuda', 0))
    tensors = [*model.parameters(), *model.buffers()]
    if not tensors or not all(v.is_cuda for v in tensors): raise ValueError('Model tensors must all be on CUDA')
    ids = torch.tensor(tokens, device='cuda')[None]; cache = None; generated = []; timings = []
    raw_eos = model.generation_config.eos_token_id
    eos = set(raw_eos if isinstance(raw_eos, list) else [raw_eos]) | {tokenizer.eos_token_id}
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(a.max_new_tokens):
            torch.cuda.synchronize(); start = time.perf_counter()
            out = model(ids, use_cache=True, past_key_values=cache)
            cache = out.past_key_values; token = out.logits[:, -1].argmax(-1)
            torch.cuda.synchronize(); timings.append(time.perf_counter() - start)
            generated.append(int(token.item())); ids = token[:, None]
            if generated[-1] in eos: break
    answer = tokenizer.decode(generated, skip_special_tokens=True); print(answer)
    result = {'completed': True, 'optimizer_updates': report['actual_optimizer_updates'], 'manifest_SHA256': identity,
              'text': answer, 'generated_token_ids': generated, 'ended_with_EOS': generated[-1] in eos,
              'first_token_seconds': timings[0], 'request_seconds_excludes_load_tokenization': sum(timings),
              'decode_after_first_tokens_per_second': (len(timings) - 1) / sum(timings[1:]) if len(timings) > 1 else None,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'fused_kernel': False, 'quality_accepted': False}
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__': main()
