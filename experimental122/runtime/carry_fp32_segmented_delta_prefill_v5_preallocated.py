"""Inference dependency extracted from validated experimental runtime."""

import hashlib, inspect

import torch

import torch.nn.functional as F

from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as q

l2norm=q.l2norm

ORIGINAL_MODELING_SHA256='f738a9eb46d1351a751f36a1cae5f7435d589e8c3218e7315182419c4c0230f1'

ORIGINAL_FUNCTION_SHA256='70c5fbcca74b52b8da19ea77b7ef1d53dcf42ecf8565d7e11b732bec298da78c'

def _carry_delta_kernel(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query = query.transpose(1, 2).contiguous().to(torch.float32)
    key = key.transpose(1, 2).contiguous().to(torch.float32)
    value = value.transpose(1, 2).contiguous().to(torch.float32)
    beta = beta.transpose(1, 2).contiguous().to(torch.float32)
    g = g.transpose(1, 2).contiguous().to(torch.float32)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    del value
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        del row, sub
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    del v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    del attn, k_beta
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else (initial_state if kwargs.get('_preserve_carried_fp32_state',False) else initial_state.to(value))
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state

def segmented_fp32_delta_prefill(query,key,value,g,beta,chunk_size=64,initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=False,max_tokens=128,**kwargs):
 assert not torch.is_grad_enabled() and chunk_size==64 and max_tokens>=64 and max_tokens%64==0
 assert kwargs.get('cu_seqlens') is None and '_preserve_carried_fp32_state' not in kwargs
 if query.shape[1]<=max_tokens:return _carry_delta_kernel(query,key,value,g,beta,chunk_size=chunk_size,initial_state=initial_state,output_final_state=output_final_state,use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,**kwargs)
 output=torch.empty((*query.shape[:-1],value.shape[-1]),device=query.device,dtype=query.dtype);state=initial_state
 for start in range(0,query.shape[1],max_tokens):
  if start:assert state.dtype==torch.float32 and state.device==query.device
  end=min(start+max_tokens,query.shape[1]);part,state=_carry_delta_kernel(query[:,start:end],key[:,start:end],value[:,start:end],g[:,start:end],beta[:,start:end],chunk_size=chunk_size,initial_state=state,output_final_state=True,use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,_preserve_carried_fp32_state=bool(start),**kwargs);output[:,start:end].copy_(part);del part
 return output,state if output_final_state else None

def bind_segmented_layer(layer):
 if hasattr(layer,'linear_attn'):
  from pathlib import Path
  assert hashlib.sha256(Path(q.__file__).read_bytes()).hexdigest()==ORIGINAL_MODELING_SHA256
  assert hashlib.sha256(inspect.getsource(q.torch_chunk_gated_delta_rule).strip().encode()).hexdigest()==ORIGINAL_FUNCTION_SHA256
  assert layer.linear_attn.chunk_gated_delta_rule is q.torch_chunk_gated_delta_rule
  layer.linear_attn.chunk_gated_delta_rule=segmented_fp32_delta_prefill

def bind_segmented_model(model):
 assert not model.training
 for layer in model.model.language_model.layers:bind_segmented_layer(layer)
