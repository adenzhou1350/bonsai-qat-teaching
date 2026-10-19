"""CPU mechanism controls. No model weights or claims of 27B training completion."""
import copy
import json
import unittest
import torch
from rotation import rotate, fixed_signs
from ternary import STE, RotatedTernary, PackedTernary, pack, unpack, quantize


class CoreControls(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(381027)
        torch.set_num_threads(1)

    def test_real_width_rotated_linear_and_inverse_embedding(self):
        width = 5120
        signs = fixed_signs(width)
        x, weight = torch.randn(3, width, dtype=torch.float64), torch.randn(7, width, dtype=torch.float64)
        xr, wr = rotate(x, signs), rotate(weight, signs)
        torch.testing.assert_close(xr @ wr.T, x @ weight.T, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(rotate(wr[[1, 6]], signs, inverse=True), weight[[1, 6]], atol=1e-12, rtol=1e-12)
        self.assertGreater(float((rotate(xr, signs) - x).abs().max()), 0.1)

    def test_packing_padding_and_corruption(self):
        for width in (1, 128, 1024, 5120):
            code = torch.randint(-1, 2, (3, width), dtype=torch.int8)
            self.assertTrue(torch.equal(unpack(pack(code), width), code))
        with self.assertRaises(ValueError):
            unpack(torch.tensor([[243]], dtype=torch.uint8), 5)

    def test_identity_STE(self):
        x = torch.randn(2, 1024, requires_grad=True)
        upstream = torch.randn_like(x)
        code, scale = quantize(x.detach())
        from ternary import unpack_grid
        (STE.apply(x, unpack_grid(code, scale)) * upstream).sum().backward()
        self.assertTrue(torch.equal(x.grad, upstream))
        self.assertEqual(scale.dtype, torch.float16)

    def test_training_complete_state_restore_next_update_and_export(self):
        student = RotatedTernary(torch.randn(8, 1024))
        optimizer = torch.optim.AdamW(student.parameters(), lr=0.01, weight_decay=0)
        x, target = torch.randn(4, 1024), torch.randn(4, 8)
        def step(model, opt):
            opt.zero_grad(set_to_none=True)
            loss = (model(x) - target).square().mean()
            loss.backward()
            self.assertTrue(bool(torch.isfinite(model.master.grad).all()))
            opt.step()
            return float(loss.detach())
        before = student.master.detach().clone()
        losses = [step(student, optimizer) for _ in range(3)]
        self.assertFalse(torch.equal(before, student.master))
        state, optstate, rng = copy.deepcopy(student.state_dict()), copy.deepcopy(optimizer.state_dict()), torch.get_rng_state()
        restored = RotatedTernary(torch.zeros(8, 1024))
        restored.load_state_dict(state)
        other = torch.optim.AdamW(restored.parameters(), lr=0.01, weight_decay=0)
        other.load_state_dict(optstate)
        torch.set_rng_state(rng)
        step(student, optimizer)
        torch.set_rng_state(rng)
        step(restored, other)
        self.assertTrue(torch.equal(student.master, restored.master))
        for key in ("step", "exp_avg", "exp_avg_sq"):
            self.assertTrue(torch.equal(optimizer.state[student.master][key], other.state[restored.master][key]))
        self.assertTrue(torch.equal(student(x), PackedTernary(student.packed())(x)))
        embedding = RotatedTernary(torch.randn(16, 1024), embedding=True)
        ids = torch.tensor([[0, 3, 15]])
        self.assertTrue(torch.equal(embedding(ids), PackedTernary(embedding.packed())(ids)))
        print(json.dumps({"small_control_loss_first": losses[0], "small_control_loss_third": losses[-1],
                          "27B_training_completed": False, "GPU_test_completed": False}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
