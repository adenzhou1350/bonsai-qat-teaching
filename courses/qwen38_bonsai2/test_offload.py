"""Small real-CUDA tests for CPU-master transfers; no full-model allocation."""
import copy
import json
import unittest
import torch
from offload import StagedTernary
from ternary import RotatedTernary, PackedTernary
from test_native import tiny_model, replace


class OffloadControls(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.fail("This check requires an explicitly admitted idle CUDA GPU")
        torch.set_num_threads(1)
        torch.manual_seed(381027)

    def test_CPU_master_CUDA_gradient_and_packed_output(self):
        original = torch.randn(8, 1024)
        staged = StagedTernary(original, compute_dtype=torch.float32)
        direct = RotatedTernary(original).cuda()
        x = torch.randn(4, 1024, device="cuda")
        y, expected = staged(x), direct(x)
        self.assertTrue(torch.equal(y, expected))
        y.square().mean().backward()
        expected.square().mean().backward()
        self.assertEqual(staged.master.grad.device.type, "cpu")
        self.assertTrue(torch.equal(staged.master.grad, direct.master.grad.cpu()))
        torch.optim.SGD(staged.parameters(), lr=1e-3).step()
        packed = PackedTernary(staged.packed(device="cuda")).cuda()
        self.assertTrue(torch.equal(staged(x), packed(x)))
        bf16 = StagedTernary(original, compute_dtype=torch.bfloat16)
        xb = x.bfloat16()
        self.assertTrue(torch.equal(bf16(xb), PackedTernary(bf16.packed(device="cuda")).cuda()(xb)))
        embedding = StagedTernary(torch.randn(16, 1024), embedding=True, compute_dtype=torch.bfloat16)
        ids = torch.tensor([[0, 3, 15]], device="cuda")
        self.assertTrue(torch.equal(embedding(ids), PackedTernary(embedding.packed(device="cuda")).cuda()(ids)))

    def test_native_dense_graph_with_all_masters_on_CPU(self):
        model = tiny_model().cuda()
        model.requires_grad_(False)
        replace(model)
        count = 0
        for name, module in list(model.named_modules()):
            if not isinstance(module, RotatedTernary):
                continue
            staged = StagedTernary(torch.zeros_like(module.master, device="cpu"),
                                  embedding=module.embedding, block=128, compute_dtype=torch.float32)
            with torch.no_grad():
                staged.master.copy_(module.master.cpu())
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent) if parent else model, leaf, staged)
            count += 1
        ids = torch.randint(0, 256, (1, 16), device="cuda")
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        logits = model(ids, use_cache=False).logits
        loss = logits.float().square().mean()
        loss.backward()
        masters = [p for p in model.parameters() if p.requires_grad]
        self.assertEqual(len(masters), 27)
        self.assertTrue(all(p.device.type == "cpu" and p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in masters))
        torch.optim.SGD(masters, lr=1e-3).step()
        print(json.dumps({"small_native_layers": 4, "CPU_masters_updated_through_CUDA": count,
                          "CUDA_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                          "27B_model_training_completed": False}))

    def test_BF16_native_cached_decode(self):
        model = tiny_model().bfloat16().cuda().eval()
        model.requires_grad_(False)
        replace(model)
        for name, module in list(model.named_modules()):
            if not isinstance(module, RotatedTernary):
                continue
            staged = StagedTernary(torch.zeros_like(module.master, device="cpu"),
                                  embedding=module.embedding, block=128, compute_dtype=torch.bfloat16)
            with torch.no_grad():
                staged.master.copy_(module.master.cpu())
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent) if parent else model, leaf, staged)
        ids = torch.randint(0, 256, (1, 9), device="cuda")
        with torch.inference_mode():
            prefill = model(ids[:, :8], use_cache=True)
            cached = model(ids[:, 8:], use_cache=True, past_key_values=prefill.past_key_values).logits[:, -1]
            direct = model(ids, use_cache=False).logits[:, -1]
        torch.testing.assert_close(cached.float(), direct.float(), atol=.03, rtol=.03)
        print(json.dumps({"BF16_native_cached_last_position_max_abs_difference": float((cached.float()-direct.float()).abs().max()),
                          "comparison_is_tolerance_based": True, "full_27B_cache_verified": False}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
