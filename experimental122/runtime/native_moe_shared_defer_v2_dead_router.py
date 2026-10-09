"""Inference dependency extracted from validated experimental runtime."""

import ast, hashlib, inspect, textwrap, types

import torch

from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as q

NATIVE_FORWARD_SHA='1d4a74496f8aa6614750abdf38fe5ca7b5db811341480f6e79de91ac92ebddad'

def own_moe_forward():
 original=q.Qwen3_5MoeSparseMoeBlock.forward
 assert hashlib.sha256(inspect.getsource(original).strip().encode()).hexdigest()==NATIVE_FORWARD_SHA
 tree=ast.parse(textwrap.dedent(inspect.getsource(original)));body=tree.body[0].body
 shared=ast.parse('shared_expert_output = self.shared_expert(hidden_states_reshaped)').body[0]
 experts=ast.parse('expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)').body[0]
 si=[i for i,n in enumerate(body) if ast.dump(n)==ast.dump(shared)];ei=[i for i,n in enumerate(body) if ast.dump(n)==ast.dump(experts)];assert si==[2] and ei==[4] and original.__closure__ is None
 node=body.pop(si[0]);body.insert(ei[0],node)
 gate=ast.parse('_, routing_weights, selected_experts = self.gate(hidden_states_reshaped)').body[0]
 gi=[i for i,n in enumerate(body) if ast.dump(n)==ast.dump(gate)];assert gi==[2]
 assert not any(isinstance(n,ast.Name) and n.id=='_' and isinstance(n.ctx,ast.Load) for statement in body[gi[0]+1:] for n in ast.walk(statement))
 body.insert(gi[0]+1,ast.parse('del _').body[0])
 namespace=dict(original.__globals__);exec(compile(ast.fix_missing_locations(tree),'<owned-native-MoE-shared-deferred>','exec'),namespace)
 owned=namespace['forward'];owned.__defaults__=original.__defaults__;owned.__kwdefaults__=original.__kwdefaults__
 return owned

def bind_moe_module(module):
 assert type(module) is q.Qwen3_5MoeSparseMoeBlock
 original=q.Qwen3_5MoeSparseMoeBlock.forward;owned=own_moe_forward()
 def forward(self,hidden_states):
  if hidden_states.shape[1]<=4096:return original(self,hidden_states)
  assert not torch.is_grad_enabled(),'Inference-only independent native computation reorder'
  return owned(self,hidden_states)
 module.forward=types.MethodType(forward,module)

def bind_moe_shared_defer_layer(layer):bind_moe_module(layer.mlp)

def bind_moe_shared_defer_model(model):
 for layer in model.model.language_model.layers:bind_moe_shared_defer_layer(layer)
