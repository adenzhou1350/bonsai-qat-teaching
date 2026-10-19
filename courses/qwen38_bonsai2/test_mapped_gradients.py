"""Compare mapped gradients with ordinary FP32 gradients and identical Adafactor updates."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import torch
from transformers.optimization import Adafactor
from mapped_gradients import MappedGradients

def optimizer(model):
    return Adafactor([p for p in model.parameters() if p.requires_grad], lr=1e-5, relative_step=False, scale_parameter=False, warmup_init=False, beta1=None, clip_threshold=1.)

def compare_state(test, left, right):
    test.assertEqual(list(left), list(right))
    for key, value in left.items():
        other = right[key]
        if torch.is_tensor(value):
            test.assertTrue(torch.equal(value, other), str(key))
        elif isinstance(value, dict):
            compare_state(test, value, other)
        else:
            test.assertEqual(value, other)

class MappedControls(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(381027)
        torch.set_num_threads(1)

    def test_CPU_accumulation_Adafactor_and_sparse_zero(self):
        original = torch.nn.Sequential(torch.nn.Linear(12, 8), torch.nn.LayerNorm(8), torch.nn.Linear(8, 3))
        mapped = copy.deepcopy(original)
        a, b = optimizer(original), optimizer(mapped)
        with tempfile.TemporaryDirectory() as temp, MappedGradients(mapped.named_parameters(), Path(temp) / 'gradients', reserve_bytes=0) as store:
            for step in range(4):
                a.zero_grad(set_to_none=True)
                store.zero_grad()
                self.assertTrue(all(bool((p.grad == 0).all()) for p in mapped.parameters()))
                for _ in range(2):
                    x = torch.randn(5, 12)
                    original(x).square().mean().backward()
                    mapped(x).square().mean().backward()
                store.assert_complete()
                for left, right in zip(original.parameters(), mapped.parameters()):
                    self.assertTrue(torch.equal(left.grad, right.grad))
                a.step()
                b.step()
                for left, right in zip(original.parameters(), mapped.parameters()):
                    self.assertTrue(torch.equal(left, right))
                    compare_state(self, a.state[left], b.state[right])
            store.zero_grad()
            self.assertTrue(all(entry['file'].stat().st_blocks == 0 for entry in store.entries))
        self.assertTrue(all(p.grad is None for p in mapped.parameters()))

    def test_missing_backward_is_rejected(self):
        layer = torch.nn.Linear(4, 2)
        with tempfile.TemporaryDirectory() as temp, MappedGradients(layer.named_parameters(), Path(temp) / 'gradients', reserve_bytes=0) as store:
            with self.assertRaises(RuntimeError):
                store.assert_complete()
            layer.weight.square().mean().backward()
            with self.assertRaises(RuntimeError):
                store.assert_complete()

    def test_CUDA_checkpoint_graph_matches_ordinary_gradients(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA case requires admitted GPU; CPU cases still apply')
        from test_native import tiny_model, replace
        from buffered_staging import BufferedStagedTernary
        from ternary import RotatedTernary
        model = tiny_model().cuda()
        model.requires_grad_(False)
        replace(model)
        for name, module in list(model.named_modules()):
            if not isinstance(module, RotatedTernary):
                continue
            staged = BufferedStagedTernary(torch.zeros_like(module.master, device='cpu'), embedding=module.embedding, block=128, compute_dtype=torch.float32)
            with torch.no_grad():
                staged.master.copy_(module.master.cpu())
            parent, _, leaf = name.rpartition('.')
            setattr(model.get_submodule(parent) if parent else model, leaf, staged)
        mapped = copy.deepcopy(model)
        named = [(name, p) for name, p in mapped.named_parameters() if p.requires_grad]
        a, b = optimizer(model), optimizer(mapped)
        for value in (model, mapped):
            value.train()
            value.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        with tempfile.TemporaryDirectory() as temp, MappedGradients(named, Path(temp) / 'gradients', reserve_bytes=0) as store:
            for step in range(3):
                a.zero_grad(set_to_none=True)
                store.zero_grad()
                for _ in range(2):
                    ids = torch.randint(0, 256, (1, 16), device='cuda')
                    lhs = model(ids, use_cache=False).logits
                    rhs = mapped(ids, use_cache=False).logits
                    self.assertTrue(torch.equal(lhs, rhs))
                    lhs.float().square().mean().backward()
                    rhs.float().square().mean().backward()
                store.assert_complete()
                for (name, left), (_, right) in zip([(n, p) for n, p in model.named_parameters() if p.requires_grad], named):
                    self.assertTrue(torch.equal(left.grad, right.grad), name)
                a.step()
                b.step()
                for left, right in zip([p for p in model.parameters() if p.requires_grad], [p for _, p in named]):
                    self.assertTrue(torch.equal(left, right))
                    compare_state(self, a.state[left], b.state[right])
        print(json.dumps({'small_native_layers': 4, 'CPU_masters': len(named), 'steps': 3, 'accumulations_per_step': 2, 'gradients_parameters_and_Adafactor_state_equal': True, 'full27B_verified': False}))

if __name__ == '__main__':
    unittest.main(verbosity=2)
