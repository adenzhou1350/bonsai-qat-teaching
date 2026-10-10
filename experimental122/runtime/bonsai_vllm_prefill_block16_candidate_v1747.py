"""Private approximate prefill candidate; decode and packed weights stay intact.

Grouped GEMM can change floating-point rounding. Execution is not quality proof.
"""
import hashlib
from pathlib import Path
import torch
from transformers.integrations.moe import _grouped_mm
import gsq_ternary_retained_factor_runtime_v17_silu_inplace as provider

AUDIT={'base_chunk_calls':0,'gate_prefill_calls':0,'active_chunks':0,'maximum_decoded_expert_chunk':0,'group_size':16,'changes_prefill_GEMM_arithmetic':True}

def block_base(bank,inputs,offsets):
    assert not torch.is_grad_enabled() and inputs.dtype==torch.bfloat16
    experts,n,k=bank.shape
    stops=offsets.detach().cpu().tolist()
    assert experts==256 and stops[-1]==len(inputs) and inputs.shape[1]==k
    result=torch.empty((len(inputs),n),device=inputs.device,dtype=torch.bfloat16)
    AUDIT['base_chunk_calls']+=1
    for begin in range(0,experts,16):
        end=min(begin+16,experts);lo=stops[begin-1] if begin else 0;hi=stops[end-1]
        if hi==lo:continue
        ids=torch.arange(begin,end,device=inputs.device,dtype=torch.int64)
        weight=bank.weights(ids)
        assert weight.shape==(end-begin,n,k) and weight.dtype==inputs.dtype
        local_ends=(offsets[begin:end]-lo).contiguous()
        result[lo:hi]=_grouped_mm(inputs[lo:hi],weight.transpose(-2,-1),local_ends)
        del weight,ids,local_ends
        AUDIT['active_chunks']+=1;AUDIT['maximum_decoded_expert_chunk']=max(AUDIT['maximum_decoded_expert_chunk'],end-begin)
    return result

def gate_base_low(bank,inputs,order,ends,fanout):
    assert not torch.is_grad_enabled() and inputs.dtype==torch.bfloat16 and fanout==8
    routed=inputs[order//fanout].contiguous()
    base=block_base(bank,routed,ends)
    low=_grouped_mm(routed,bank.lora_a.to(inputs.dtype).transpose(-2,-1),ends)
    AUDIT['gate_prefill_calls']+=1
    return base,low

def install():
    expected='gsq_ternary_retained_factor_runtime_v17_silu_inplace.py'
    path=Path(provider.__file__);assert path.name==expected
    originals=(provider.indexed_gate_base_low_projection,provider.bounded_ternary_base_projection)
    provider.indexed_gate_base_low_projection=gate_base_low
    provider.bounded_ternary_base_projection=block_base
    return {'provider_source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'group_size':16,'changes_prefill_GEMM_arithmetic':True,'persistent_new_CUDA_weight_storage_bytes':0},originals

def restore(originals):
    provider.indexed_gate_base_low_projection,provider.bounded_ternary_base_projection=originals
