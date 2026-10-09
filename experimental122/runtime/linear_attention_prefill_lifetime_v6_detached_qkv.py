"""Inference dependency extracted from validated experimental runtime."""

from native_bounded_causal_conv_silu_v1 import bounded_causal_conv_silu

import ast, hashlib, inspect, textwrap, types

from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as q

ORIGINAL_FORWARD_SHA256='ce12238c36d873531ebb2680767fc75058db9077e9c9185093d2a4caf5484f0a'

def bind_linear_prefill_lifetime_layer(layer):
 if not hasattr(layer,'linear_attn'):return
 original=q.Qwen3_5MoeGatedDeltaNet.forward
 assert hashlib.sha256(inspect.getsource(original).strip().encode()).hexdigest()==ORIGINAL_FORWARD_SHA256
 target=layer.linear_attn
 assert type(target) is q.Qwen3_5MoeGatedDeltaNet and target.forward.__func__ is original
 for module in (target.in_proj_z,target.in_proj_b,target.in_proj_a,target.conv1d):
  assert not module._forward_hooks and not module._forward_pre_hooks
 tree=ast.parse(textwrap.dedent(inspect.getsource(original)).strip());function=tree.body[0]
 class Conv(ast.NodeTransformer):
  count=0
  def visit_Assign(self,node):
   expected=ast.parse('mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :mixed_qkv.shape[-1]])').body[0]
   if ast.dump(node)==ast.dump(expected):
    self.count+=1;return ast.copy_location(ast.parse('mixed_qkv = bounded_causal_conv_silu(self.conv1d, mixed_qkv)').body[0],node)
   return self.generic_visit(node)
 conv=Conv();conv.visit(tree);assert conv.count==1
 value_at=[i for i,n in enumerate(function.body) if isinstance(n,ast.Assign) and ast.unparse(n)=='value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)']
 assert len(value_at)==1
 detach=ast.parse('if seq_len > 4096:\n assert not torch.is_grad_enabled()\n assert self.num_v_heads // self.num_k_heads > 1\n query, key, value = [v.contiguous() for v in (query, key, value)]\n assert all(v.is_contiguous() and v.untyped_storage().data_ptr() != mixed_qkv.untyped_storage().data_ptr() for v in (query,key,value))').body[0]
 function.body.insert(value_at[0]+1,detach)
 repeat_at=[i for i,n in enumerate(function.body) if isinstance(n,ast.If) and ast.unparse(n.test)=='self.num_v_heads // self.num_k_heads > 1']
 assert len(repeat_at)==1
 assert [ast.unparse(n) for n in function.body[repeat_at[0]].body]==['query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)', 'key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)']
 retire=ast.parse('if seq_len > 4096:\n assert all(v.untyped_storage().data_ptr() != mixed_qkv.untyped_storage().data_ptr() for v in (query,key,value))\n del mixed_qkv').body[0]
 function.body.insert(value_at[0]+2,retire)
 z_at=[i for i,n in enumerate(function.body) if isinstance(n,ast.Assign) and ast.unparse(n)=='z = self.in_proj_z(hidden_states)']
 assert len(z_at)==1;i=z_at[0];z_statements=function.body[i:i+2]
 assert ast.unparse(z_statements[1])=='z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)'
 # Short inputs and decode retain the original order. Long inputs execute
 # these exact two complete projections after the delta rule and cache update.
 function.body[i:i+2]=[ast.If(test=ast.parse('seq_len <= 4096',mode='eval').body,body=z_statements,orelse=[])]
 at=[i for i,n in enumerate(function.body) if isinstance(n,ast.Assign) and ast.unparse(n)=='core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)']
 assert len(at)==1;retired={'query','key','value','mixed_qkv','b','a','beta','g'}
 assert not retired & {n.id for statement in function.body[at[0]:] for n in ast.walk(statement) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
 release=ast.parse('if seq_len > 4096:\n assert not torch.is_grad_enabled()\n del query, key, value, b, a, beta, g').body[0]
 release.body.extend(z_statements)
 function.body.insert(at[0],release);ast.fix_missing_locations(tree)
 private=dict(original.__globals__);private['bounded_causal_conv_silu']=bounded_causal_conv_silu;exec(compile(tree,__file__,'exec'),private)
 target.forward=types.MethodType(private['forward'],target)

def bind_linear_prefill_lifetime_model(model):
 assert not model.training
 for layer in model.model.language_model.layers:bind_linear_prefill_lifetime_layer(layer)
