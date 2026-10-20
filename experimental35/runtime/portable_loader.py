"""Portable assembly from the recorded R35 loader; admission is in infer.py.

Expert arithmetic, retained tensor checks, model construction and CUDA dtype
statements are unchanged. No training or private experiment-directory access.
"""
import gc,json
from pathlib import Path
import torch
from transformers import AutoConfig,AutoTokenizer,GenerationConfig,Qwen3_5MoeForConditionalGeneration
from balanced35_recovery_common_v1 import backend,sha,tensor_sha
from gsq35_balanced_factor_runtime_v2_mm_out import PackedGSQFactorExperts
def bank_rows(directory, report):
    rows = {}
    for row in report['matrices']:
        assert row['packing_format'] == 'own-gsq-symmetric-expert-bank-v1' and row['attr'] in ('gate_up_proj', 'down_proj')
        li = int(row['module'].split('.')[1])
        assert sha(Path(directory) / row['file']) == row['sha256']
        assert row['attr'] not in rows.setdefault(li, {})
        rows[li][row['attr']] = row
    assert set(rows) == set(range(40)) and all((set(v) == {'gate_up_proj', 'down_proj'} for v in rows.values()))
    return rows

def read_banks(directory, rows):
    banks = {}
    for name, row in rows.items():
        bank = torch.load(Path(directory) / row['file'], map_location='cpu', weights_only=True)
        assert bank['format'] == 'own_gsq_symmetric_expert_bank_v1' and bank['shape'] == row['shape'] and (len(bank['states']) == 256)
        _, n, k = bank['shape']
        for state in bank['states']:
            assert state['format'] == 'own_contiguous_bit_cpu_v1' and state['nbits'] == 2 and (state['group_size'] == 128) and (state['shape'] == [n, k]) and ('weight' not in state) and (state['lora_scale'] == 0.125)
            assert state['lora_A'].dtype == state['lora_B'].dtype == torch.bfloat16 and state['lora_A'].shape == (16, k) and (state['lora_B'].shape == (n, 16))
            assert torch.isfinite(state['lora_A']).all() and torch.isfinite(state['lora_B']).all()
        banks[name] = bank
    return banks

def retained(directory, report):
    values = torch.load(Path(directory) / 'retained.pt', map_location='cpu', weights_only=True)
    assert set(values) == set(report['retained_tensor_sha256'])
    assert sum((v.numel() * v.element_size() for v in values.values())) == report['retained_tensor_bytes']
    for name, value in values.items():
        assert '.mlp.experts.' not in name and tensor_sha(value) == report['retained_tensor_sha256'][name]
    return values

def load_packed(directory, report):
    directory = Path(directory)
    backend()
    cfg = AutoConfig.from_pretrained(directory, local_files_only=True)
    cfg._attn_implementation = 'sdpa'
    cfg.text_config._attn_implementation = 'sdpa'
    cfg.text_config._experts_implementation = 'grouped_mm'
    with torch.device('meta'):
        model = Qwen3_5MoeForConditionalGeneration(cfg)
    rows = bank_rows(directory, report)
    language = model.model.language_model
    for li in range(40):
        banks = read_banks(directory, rows[li])
        language.layers[li].mlp.experts = PackedGSQFactorExperts(banks)
        del banks
        gc.collect()
    values = retained(directory, report)
    missing, extra = model.load_state_dict({n: v.to('cuda') for n, v in values.items()}, strict=False, assign=True)
    assert not extra
    assert all((p.device.type == 'cuda' for n, p in model.named_parameters() if n.startswith(('model.language_model.', 'lm_head.'))))
    assert all((v.device.type == 'cuda' for layer in language.layers for v in layer.mlp.experts.buffers()))
    language.rotary_emb = type(language.rotary_emb)(cfg.text_config, device=torch.device('cuda'))
    model.generation_config = GenerationConfig.from_pretrained(directory, local_files_only=True)
    model.requires_grad_(False)
    model.eval()
    del values
    gc.collect()
    return (AutoTokenizer.from_pretrained(directory, local_files_only=True), model, report)
