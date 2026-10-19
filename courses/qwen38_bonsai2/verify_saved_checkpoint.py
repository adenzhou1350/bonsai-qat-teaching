"""Independently restore saved masters, optimizer and RNG; compare the matching packed export."""
import argparse
import gc
import json
import random
import time
from pathlib import Path
import torch
from transformers.optimization import Adafactor
from buffered_staging import BufferedStagedTernary
from control import packed_reload, sha, tensor_sha
from mapped_gradients import memory_admission
from offload import load_model
from recovery_checkpoint import load_checkpoint
from verify_source import check_file

def nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(nested_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(nested_equal(a, b) for a, b in zip(left, right))
    return left == right

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'data', 'checkpoint', 'packed', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--checkpoint-manifest-sha256', required=True)
    p.add_argument('--packed-manifest-sha256', required=True)
    p.add_argument('--teacher-cache-sha256', required=True)
    p.add_argument('--expected-step', type=int, required=True)
    a = p.parse_args()
    if not 1 <= a.expected_step <= 32 or torch.cuda.device_count() != 1:
        raise ValueError('One admitted GPU and a bounded pilot checkpoint required')
    a.output.mkdir(parents=True, exist_ok=False)
    report = {'format': 'qwen38-independent-saved-checkpoint-verification-v1',
              'source_control_updates': a.expected_step, 'new_optimizer_updates': 0,
              'quality_accepted': False, 'saved_artifacts_modified': False, 'passed': False,
              'next_optimizer_update_after_restore_verified': False,
              'original_live_pre_checkpoint_reference_available': False}
    started = time.monotonic()
    def save(stage, **values):
        report.update(state=stage, seconds=time.monotonic()-started, **values)
        part = a.output / 'progress.tmp'
        part.write_text(json.dumps(report, indent=2), encoding='utf-8')
        part.replace(a.output / 'progress.json')
    if sha(a.checkpoint / 'manifest.json') != a.checkpoint_manifest_sha256 or sha(a.packed / 'manifest.json') != a.packed_manifest_sha256:
        raise ValueError('Saved artifact manifest identity changed')
    checkpoint_manifest = json.loads((a.checkpoint / 'manifest.json').read_text())
    packed_manifest = json.loads((a.packed / 'manifest.json').read_text())
    data = json.loads(a.data.read_text())
    spec = json.loads(Path(__file__).with_name('source-files.json').read_text())
    expected = {'step': a.expected_step, 'source_revision': spec['revision'], 'data_SHA256': sha(a.data),
                'teacher_cache_SHA256': a.teacher_cache_sha256,
                'optimizer': 'Adafactor_beta1None_relativeFalse_scaleFalse',
                'base_learning_rate': 1e-5, 'warmup_steps': 4,
                'gradient_storage': 'local_file_mmap_FP32', 'rotation_seed': 381027,
                'rotation_block': 1024, 'group': 128}
    if spec['revision'] != '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0' or checkpoint_manifest['metadata'] != expected:
        raise ValueError('Expected source/data/optimizer/rotation metadata differs')
    if not checkpoint_manifest['complete'] or len(checkpoint_manifest['masters']) != 402 or packed_manifest['actual_optimizer_updates'] != a.expected_step:
        raise ValueError('Incomplete or mismatched artifacts')
    save('verifying_original_basis_files', memory_admission=memory_admission(26869760000, 248320*5120),
         checkpoint_manifest_SHA256=a.checkpoint_manifest_sha256, packed_manifest_SHA256=a.packed_manifest_sha256,
         gradient_scratch_allocated=False, checkpoint_tensor_and_state_bytes=checkpoint_manifest['written_tensor_and_state_bytes'])
    for index, row in enumerate(spec['files']):
        check_file(a.model / row['name'], row)
        save('verifying_original_basis_files', verified_original_files=index+1)
    torch.set_num_threads(8)
    torch.manual_seed(381027)
    def loaded(name, count):
        if count % 16 == 0:
            save('reconstructing_original_rotated_CPU_basis', matrices_loaded=count)
    student, cfg = load_model(a.model, torch.device('cuda', 0), quantized=True, on_loaded=loaded, staged_type=BufferedStagedTernary)
    if sum(p.numel() for p in student.parameters() if p.requires_grad) != 26869760000:
        raise ValueError('Master scope differs')
    optimizer = Adafactor([p for p in student.parameters() if p.requires_grad], lr=1e-5,
                         relative_step=False, scale_parameter=False, warmup_init=False,
                         beta1=None, clip_threshold=1., weight_decay=0.)
    save('restoring_saved402_FP32_masters_optimizer_and_RNG')
    restored_manifest = load_checkpoint(student, optimizer, a.checkpoint, expected)
    state = torch.load(a.checkpoint / 'state.pt', map_location='cpu', weights_only=True)
    if not nested_equal(optimizer.state_dict(), state['optimizer']):
        raise ValueError('Restored optimizer state differs')
    if not torch.equal(torch.get_rng_state(), state['torch_RNG']) or random.getstate() != state['python_RNG']:
        raise ValueError('Restored CPU/Python RNG differs')
    if state['CUDA_RNG'] is not None and not nested_equal(torch.cuda.get_rng_state_all(), state['CUDA_RNG']):
        raise ValueError('Restored CUDA RNG differs')
    del state
    kept = {n: tensor_sha(p) for n, p in student.named_parameters() if not p.requires_grad}
    if kept != packed_manifest['retained_state_hashes']:
        raise ValueError('Retained parameters differ from matching packed artifact')
    save('all_saved_masters_optimizer_RNG_and_retained_state_verified',
         checkpoint_decoded_masters_exact_raw_SHA256_verified=len(restored_manifest['masters']),
         optimizer_state_equal_saved=True, Torch_Python_CUDA_RNG_equal_saved=True,
         retained_state_equal_packed_export=True)
    ids = torch.tensor(data['rows']['validation'][0]['input_ids'], device='cuda')[None]
    student.train()
    with torch.no_grad():
        reference = student.lm_head(student.model(ids, use_cache=False).last_hidden_state[:, -1:]).cpu()
    reference_sha = tensor_sha(reference)
    del optimizer, student
    gc.collect()
    torch.cuda.empty_cache()
    save('loading_matching_packed_export_for_forward_comparison', restored_last_position_logits_SHA256=reference_sha)
    packed, _ = packed_reload(a.packed, torch.device('cuda', 0))
    if not all(t.is_cuda for t in [*packed.parameters(), *packed.buffers()]):
        raise ValueError('Packed model must be entirely on CUDA')
    with torch.no_grad():
        actual = packed.lm_head(packed.model(ids, use_cache=False).last_hidden_state[:, -1:]).cpu()
    equal = torch.equal(reference, actual)
    save('packed_forward_compared_to_restored_checkpoint',
         packed_last_position_logits_SHA256=tensor_sha(actual),
         restored_checkpoint_and_packed_last_position_logits_bitwise_equal=equal,
         maximum_absolute_logit_difference=float((reference.float()-actual.float()).abs().max()))
    if not equal:
        raise ValueError('Restored checkpoint and matching packed forward differ')
    report['passed'] = True
    save('independent_checkpoint_restore_and_matching_packed_forward_verified')
    (a.output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))

if __name__ == '__main__':
    main()
