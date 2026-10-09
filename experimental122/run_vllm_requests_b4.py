"""Experimental native4095 vLLM text requests with up to four active sequences.

One queued generate call; short prompts, greedy cap64, no HTTP or vision.
Request timing includes prefill and queue refill, excludes model loading.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New output directory.')
    parser.add_argument('--decode-graph', action='store_true')
    parser.add_argument('--audit-graph-replays', action='store_true', help='Diagnostic instrumentation.')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--max-model-len', type=int, choices=(1536,), default=1536)
    parser.add_argument('--kv-cache-mib', type=int, default=1024)
    args = parser.parse_args()
    assert 1 <= args.max_new_tokens <= 64 and args.kv_cache_mib >= 1024
    assert not args.audit_graph_replays or args.decode_graph

    import gc
    import json
    import os
    import sys
    import time
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['BONSAI_BATCHED_PREFILL'] = '1'
    package = Path(__file__).resolve().parent
    sys.path.insert(0, str(package / 'runtime'))
    from bonsai_native122_vllm_request_prefill_v1662 import register_model, ARCHITECTURE, LOAD_FORMAT, MIXED_AUDIT
    from bonsai_native122_vllm_dense_v1576 import NAME
    from bonsai_native122_vllm_experts_v1564 import sha
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    assert torch.cuda.device_count() == 1 and '5090' in torch.cuda.get_device_name()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    policy = json.loads((package / 'vllm-b4-provenance.json').read_text(encoding='utf-8'))['file_sha256']
    assert len(policy) == 58 and all(sha(package.parent / name) == digest for name, digest in policy.items())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'source-policy.json').write_text(json.dumps(policy, indent=2))
    artifact = args.model.resolve()
    rows = [json.loads(line) for line in args.requests.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert rows and all(isinstance(row, dict) and isinstance(row.get('prompt'), str) and row['prompt'].strip() for row in rows)
    tokenizer = AutoTokenizer.from_pretrained(artifact, local_files_only=True)
    inputs = [tokenizer.encode(tokenizer.apply_chat_template(
        [{'role': 'user', 'content': row['prompt']}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False), add_special_tokens=False) for row in rows]
    assert max(map(len, inputs)) <= 128
    assert max(map(len, inputs)) + args.max_new_tokens < args.max_model_len
    eos = json.loads((artifact / 'generation_config.json').read_text())['eos_token_id']
    eos = set(eos if isinstance(eos, list) else [eos])
    assert eos and None not in eos
    started = time.monotonic()
    report = {'completed': False, 'stage': 'starting', 'requests': [],
              'artifact_manifest_sha256': sha(artifact / 'summary.json'),
              'CPU_weight_offload': False, 'eos_respected': True,
              'max_new_tokens': args.max_new_tokens, 'max_model_len': args.max_model_len,
              'max_num_seqs': 4, 'max_num_batched_tokens': 256,
              'KV_cache_budget_bytes': args.kv_cache_mib * 1024**2,
              'decode_graph': args.decode_graph, 'graph_replay_audit': args.audit_graph_replays,
              'speedup_claimed': False,
              'limits': 'Experimental vLLM0.24 one5090 text-only queued requests, maximum128 input tokens and cap64. Each prompt retains its own prefill GEMM shape; decode preserves scalar native arithmetic. Eager prefill reads sequence boundaries once per context. Batch timing includes prefill/refill/output collection, excludes model load/tokenization/JSON; first inference may include JIT. No per-request latency inferred from batch time. Allocator peaks/end observations are not full-process NVML peaks. No HTTP, vision, long-input or sustained-service claim.'}

    def save(stage):
        report.update(stage=stage, wall_seconds=time.monotonic() - started)
        temporary = output / 'result.tmp'
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        temporary.replace(output / 'result.json')

    replay_count = {'calls': 0}
    original_replay = torch.cuda.CUDAGraph.replay

    def counted_replay(graph):
        replay_count['calls'] += 1
        return original_replay(graph)

    if args.audit_graph_replays:
        torch.cuda.CUDAGraph.replay = counted_replay
    llm = None
    try:
        register_model()
        save('loading_full_vllm_B4_engine')
        compilation = {'mode': 0, 'cudagraph_mode': 'FULL_DECODE_ONLY',
                       'cudagraph_capture_sizes': [1, 2, 4]} if args.decode_graph else 0
        llm = LLM(model=str(artifact), tokenizer=str(artifact), dtype='bfloat16',
                  quantization=NAME, load_format=LOAD_FORMAT,
                  model_loader_extra_config={'weight_proof': str(output / 'weight-proof.json')},
                  hf_overrides={'architectures': [ARCHITECTURE], 'quantization_config': {
                      'quant_method': NAME, 'artifact': str(artifact), 'compact_model_control_only': True}},
                  tensor_parallel_size=1, distributed_executor_backend='uni',
                  max_num_seqs=4, max_model_len=args.max_model_len, max_num_batched_tokens=256,
                  enforce_eager=not args.decode_graph, compilation_config=compilation,
                  kv_cache_memory_bytes=args.kv_cache_mib * 1024**2, enable_prefix_caching=False,
                  gpu_memory_utilization=.97, cpu_offload_gb=0, seed=1614)
        proof = json.loads((output / 'weight-proof.json').read_text())
        assert proof['all361_retained_input_and_loaded_tensor_hashes_exact']
        report.update(actual_full_vllm_engine_constructed=True,
                      unique_CUDA_model_storage_bytes=proof['unique_CUDA_model_storage_bytes'],
                      weight_proof_sha256=sha(output / 'weight-proof.json'),
                      model_load_seconds=time.monotonic() - started)
        entries = [{'mode': str(wrapper.runtime_mode), 'batch_descriptor': str(key),
                    'actual_CUDAGraph_created': value.cudagraph is not None}
                   for wrapper in list(CUDAGraphWrapper._all_instances)
                   for key, value in wrapper.concrete_cudagraph_entries.items()]
        report['actual_capture_entries'] = entries
        if args.decode_graph:
            assert len(entries) == 3 and all(value['actual_CUDAGraph_created'] for value in entries)
        save('generating_queued_requests')
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = replay_count['calls']
        begin = time.monotonic()
        outputs = llm.generate([{'prompt_token_ids': ids} for ids in inputs],
                               SamplingParams(temperature=0, max_tokens=args.max_new_tokens,
                                              ignore_eos=False), use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - begin
        assert len(outputs) == len(inputs)
        for index, (request, ids) in enumerate(zip(outputs, inputs)):
            assert list(request.prompt_token_ids) == ids
            value = request.outputs[0]
            tokens = list(value.token_ids)
            assert 1 <= len(tokens) <= args.max_new_tokens
            report['requests'].append({'request_index': index, 'input_tokens': len(ids),
                                       'output_tokens': len(tokens), 'generated_token_ids': tokens,
                                       'text': value.text, 'eos_reached': tokens[-1] in eos,
                                       'finish_reason': value.finish_reason})
        emitted = sum(row['output_tokens'] for row in report['requests'])
        free, total = torch.cuda.mem_get_info()
        count = replay_count['calls'] - before
        if args.audit_graph_replays:
            assert count > 0
        report.update(total_request_seconds=elapsed, emitted_tokens=emitted,
                      aggregate_output_tokens_per_second=emitted / elapsed,
                      actual_CUDAGraph_replays=count if args.audit_graph_replays else None,
                      mixed_decode_audit=MIXED_AUDIT,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      device_used_bytes_at_observation=total - free, device_total_bytes=total)
        assert all(sha(package.parent / name) == digest for name, digest in policy.items())
        report['completed'] = True
        save('completed')
        print(json.dumps({'completed': True, 'requests': len(outputs), 'emitted_tokens': emitted,
                          'total_request_seconds': elapsed, 'graph_replays': report['actual_CUDAGraph_replays']}))
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
            del llm
            gc.collect()
        torch.cuda.CUDAGraph.replay = original_replay


if __name__ == '__main__':
    main()
