"""Matched original-BF16 prompt reference with CPU weight offload; no training."""
import argparse
import gc
import json
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from control import sha
from offload import load_model
from verify_source import check_file

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--prompts', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--quantized-diagnostics', type=Path, required=True)
    p.add_argument('--quantized-diagnostic-sha256', required=True)
    a = p.parse_args()
    cases = json.loads(a.prompts.read_text(encoding='utf-8'))
    if len(cases) != 8 or len({row['id'] for row in cases}) != 8:
        raise ValueError('Expected the fixed eight distinct diagnostic prompts')
    if any(not 1 <= row['max_new_tokens'] <= 32 for row in cases):
        raise ValueError('Bounded short generations only')
    if torch.cuda.device_count() != 1:
        raise ValueError('Expose one admitted GPU')
    a.output.mkdir(parents=True, exist_ok=False)
    report = {'format': 'qwen38-original-BF16-eight-prompt-reference-v1', 'quality_accepted': False,
              'formal_capability_benchmark': False, 'training_or_weight_changes': False,
              'prompts_SHA256': sha(a.prompts), 'cases': [], 'completed': False,
              'limitation': 'Eight development prompts with short greedy outputs; exact text checks are instruction probes, not a capability-retention score. Original BF16 with CPU weight offload; timings cannot be compared as GPU-only speed to the packed all-CUDA run.'}
    def save(stage):
        report['state'] = stage
        temp = a.output / 'progress.tmp'
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(a.output / 'progress.json')
    save('verifying_original_source_files')
    torch.set_num_threads(8)
    if sha(a.quantized_diagnostics) != a.quantized_diagnostic_sha256:
        raise ValueError('Matched quantized diagnostic identity changed')
    quantized = json.loads(a.quantized_diagnostics.read_text(encoding='utf-8'))
    if not quantized['completed'] or quantized['prompts_SHA256'] != sha(a.prompts):
        raise ValueError('Prompt set differs from completed quantized diagnostic')
    matched = {r['id']: r for r in quantized['cases']}
    if set(matched) != {r['id'] for r in cases}:
        raise ValueError('Prompt IDs differ')
    spec = json.loads(Path(__file__).with_name('source-files.json').read_text())
    if spec['revision'] != '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0':
        raise ValueError('Pinned source revision changed')
    proofs = []
    for index, row in enumerate(spec['files']):
        proofs.append(check_file(a.model / row['name'], row))
        report['verified_original_files'] = index + 1
        save('verifying_original_source_files')
    (a.output / 'original-proof.json').write_text(json.dumps({'passed': True, 'revision': spec['revision'], 'files': proofs}, indent=2))
    save('loading_original_BF16_CPU_offload_model')
    model, _ = load_model(a.model, torch.device('cuda', 0), quantized=False)
    tensors = [*model.parameters(), *model.buffers()]
    if not tensors or not any(v.device.type == 'cpu' for v in tensors):
        raise ValueError('Expected original BF16 weights with explicit CPU offload')
    gpu_stores = {v.untyped_storage().data_ptr(): v.untyped_storage().nbytes() for v in tensors if v.is_cuda}
    cpu_stores = {v.untyped_storage().data_ptr(): v.untyped_storage().nbytes() for v in tensors if v.device.type == 'cpu'}
    report.update(original_source_revision=spec['revision'], original_BF16_reference=True,
                  matched_quantized_diagnostic_SHA256=sha(a.quantized_diagnostics),
                  matched_quantized_packed_manifest_SHA256=quantized['packed_manifest_SHA256'],
                  CPU_weight_offload_during_inference=True,
                  all_model_tensors_on_CUDA=all(v.is_cuda for v in tensors),
                  model_unique_CUDA_storage_bytes=sum(gpu_stores.values()),
                  model_unique_CPU_storage_bytes=sum(cpu_stores.values()),
                  GPU_only_speed_comparison_claimed=False)
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    raw_eos = model.generation_config.eos_token_id
    eos = set(raw_eos if isinstance(raw_eos, list) else [raw_eos])
    eos.add(tokenizer.eos_token_id)
    for row in cases:
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) > 128:
            raise ValueError('Short prompts only')
        if tokens != matched[row['id']]['input_token_IDs']:
            raise ValueError('Input token IDs differ from matched quantized prompt')
        ids = torch.tensor(tokens, device='cuda')[None]
        cache = None
        generated, timings = [], []
        first = {}
        torch.cuda.reset_peak_memory_stats()
        save('generating_' + row['id'])
        with torch.inference_mode():
            for index in range(row['max_new_tokens']):
                torch.cuda.synchronize()
                start = time.perf_counter()
                out = model(ids, use_cache=True, past_key_values=cache)
                cache = out.past_key_values
                token = out.logits[:, -1].argmax(-1)
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - start)
                generated.append(int(token.item()))
                if index == 0:
                    probabilities = out.logits[0, -1].float().softmax(-1)
                    scores, indices = probabilities.topk(5)
                    first = {'top5_token_IDs': indices.tolist(), 'top5_probabilities': scores.tolist(),
                             'EOS_probability': sum(float(probabilities[t]) for t in eos if t is not None)}
                    del probabilities, scores, indices
                ids = token[:, None]
                if generated[-1] in eos:
                    break
        text = tokenizer.decode(generated, skip_special_tokens=True)
        expected = row.get('accepted_exact_texts')
        normalized = text.strip().strip('。.!！ \t\r\n')
        result = {'id': row['id'], 'prompt': row['prompt'], 'input_token_IDs': tokens,
                  'generated_token_IDs': generated, 'text': text,
                  'ended_with_EOS': generated[-1] in eos, 'first_token': first,
                  'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'request_seconds_excludes_load_tokenization': sum(timings),
                  'first_token_seconds': timings[0],
                  'decode_after_first_tokens_per_second': (len(timings)-1)/sum(timings[1:]) if len(timings)>1 else None,
                  'accepted_exact_texts': expected,
                  'normalized_exact_instruction_probe_passed': normalized in expected if expected else None}
        result['same_prompt_input_IDs_as_quantized_reference'] = True
        report['cases'].append(result)
        del out, cache, ids, token
        gc.collect()
        torch.cuda.empty_cache()
        save('completed_' + row['id'])
    report['completed'] = True
    save('all_eight_diagnostic_prompts_completed')
    (a.output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'completed': True, 'cases': len(report['cases']), 'quality_accepted': False}))

if __name__ == '__main__':
    main()
