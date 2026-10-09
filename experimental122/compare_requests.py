"""Compare complete resident CLI results before interpreting throughput."""
import argparse,json
from pathlib import Path


def describe(data):
 rows=data['requests'];timings=data.get('groups') or rows
 seconds=sum(v['request_seconds'] for v in timings)
 tokens=sum(v['output_tokens'] for v in rows)
 return {'request_count':len(rows),'output_tokens':tokens,'request_seconds':seconds,
  'aggregate_end_to_end_output_tokens_per_second':tokens/seconds,
  'peak_allocated_bytes':max(v['peak_allocated_bytes'] for v in timings),
  'peak_reserved_bytes':max(v['peak_reserved_bytes'] for v in timings),
  'request_end_device_used_bytes_max':max(v['device_used_bytes_at_observation'] for v in timings)}


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--baseline',required=True,type=Path);p.add_argument('--candidate',required=True,type=Path)
 a=p.parse_args();baseline=json.loads(a.baseline.read_text(encoding='utf-8'));candidate=json.loads(a.candidate.read_text(encoding='utf-8'))
 for data in (baseline,candidate):
  assert data['completed'] and data['eos_respected'] and not data['CPU_weight_offload']
  assert data['requests'] and [r['request_index'] for r in data['requests']]==list(range(len(data['requests'])))
  assert all(r['output_tokens']==len(r['generated_token_ids']) for r in data['requests'])
 for key in ('artifact_manifest_sha256','max_new_tokens','max_cache_len','unique_CUDA_text_storage_bytes'):
  assert baseline[key]==candidate[key],('Different experiment setting',key)
 assert len(baseline['requests'])==len(candidate['requests'])
 for original,new in zip(baseline['requests'],candidate['requests']):
  for key in ('input_tokens','generated_token_ids','eos_reached'):
   assert original[key]==new[key],('Request differs',original['request_index'],key)
 before=describe(baseline);after=describe(candidate)
 print(json.dumps({'all_emitted_token_ids_equal':True,'baseline':before,'candidate':after,
  'aggregate_end_to_end_throughput_ratio':before['request_seconds']/after['request_seconds'],
  'limits':'Request/group timing excludes model load and tokenization/JSON writing. Single fixed-order run comparison; not a randomized service benchmark. Memory maxima are allocator peaks or request-end observations, not full-device sampled peaks.'},indent=2))


if __name__=='__main__':main()
