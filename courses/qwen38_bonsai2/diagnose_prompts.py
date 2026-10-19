"""Small deterministic prompt diagnostics for a teaching packed model; no training."""
import argparse
import gc
import json
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from control import packed_reload, sha

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packed', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--prompts', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference-preview', type=Path)
    a = p.parse_args()
    cases = json.loads(a.prompts.read_text(encoding='utf-8'))
    if len(cases) != 8 or len({row['id'] for row in cases}) != 8:
        raise ValueError('Expected the fixed eight distinct diagnostic prompts')
    if any(not 1 <= row['max_new_tokens'] <= 32 for row in cases):
        raise ValueError('Bounded short generations only')
    if torch.cuda.device_count() != 1:
        raise ValueError('Expose one admitted GPU')
    a.output.mkdir(parents=True, exist_ok=False)
    report = {'format': 'qwen38-eight-prompt-diagnostic-v1', 'quality_accepted': False,
              'formal_capability_benchmark': False, 'training_or_weight_changes': False,
              'prompts_SHA256': sha(a.prompts), 'cases': [], 'completed': False,
              'limitation': 'Eight development prompts with short greedy outputs; exact text checks are instruction probes, not a capability-retention score. Packed expansion reference, not a fused serving benchmark.'}
    def save(stage):
        report['state'] = stage
        temp = a.output / 'progress.tmp'
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(a.output / 'progress.json')
    save('loading_same_packed_artifact')
    torch.set_num_threads(4)
    model, manifest = packed_reload(a.packed, torch.device('cuda', 0))
    tensors = [*model.parameters(), *model.buffers()]
    if not tensors or not all(value.is_cuda for value in tensors):
        raise ValueError('Every model tensor must be on CUDA')
    stores = {v.untyped_storage().data_ptr(): v.untyped_storage().nbytes() for v in tensors}
    report.update(source_control_updates=manifest['actual_optimizer_updates'],
                  packed_manifest_SHA256=sha(a.packed / 'manifest.json'),
                  all_model_tensors_on_CUDA=True, CPU_weight_offload_during_inference=False,
                  model_unique_CUDA_storage_bytes=sum(stores.values()), fused_kernel=False)
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    raw_eos = model.generation_config.eos_token_id
    eos = set(raw_eos if isinstance(raw_eos, list) else [raw_eos])
    eos.add(tokenizer.eos_token_id)
    for row in cases:
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) > 128:
            raise ValueError('Short prompts only')
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
        if row['id'] == 'quantization_zh' and a.reference_preview:
            ref = json.loads(a.reference_preview.read_text(encoding='utf-8'))
            if ref['packed_manifest_SHA256'] != report['packed_manifest_SHA256']:
                raise ValueError('Reference preview artifact differs')
            result['same_prompt_token_IDs_equal_previous_preview'] = generated == ref['generated_token_IDs']
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
