"""Tiny native dense Qwen graph control, not a loaded or trained 27B model."""
import copy
import json
import unittest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
from policy import quantized_matrix
from ternary import RotatedTernary, PackedTernary


def tiny_model():
    cfg = Qwen3_5TextConfig(vocab_size=256, hidden_size=128, intermediate_size=256,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=64,
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=64, linear_value_head_dim=64,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000000,
                         "partial_rotary_factor": 0.25, "mrope_section": [2, 2, 4]})
    cfg._attn_implementation = "eager"
    qwen.FusedRMSNormGated = None
    qwen.causal_conv1d_fn = qwen.causal_conv1d_update = None
    qwen.chunk_gated_delta_rule = qwen.fused_recurrent_gated_delta_rule = None
    return qwen.Qwen3_5ForCausalLM(cfg).eval()


def replace(model, packed=False):
    count = 0
    for name, module in list(model.named_modules()):
        policy_name = name if name == "lm_head" else name.replace("model.", "model.language_model.", 1)
        if not quantized_matrix(policy_name + ".weight"):
            continue
        if packed:
            new = PackedTernary(module.packed())
        else:
            new = RotatedTernary(module.weight, embedding=name.endswith("embed_tokens"), block=128)
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, new)
        count += 1
    return count


class NativeControl(unittest.TestCase):
    def test_dense_graph_KD_update_and_export_reload(self):
        torch.manual_seed(381027)
        torch.set_num_threads(1)
        teacher = tiny_model()
        student = copy.deepcopy(teacher)
        student.requires_grad_(False)
        count = replace(student)
        self.assertEqual(count, 27)
        named = [(n, p) for n, p in student.named_parameters() if p.requires_grad]
        self.assertEqual(len(named), count)
        kept = {n: p.detach().clone() for n, p in student.named_parameters() if not p.requires_grad}
        optimizer = torch.optim.AdamW([p for _, p in named], lr=1e-3, weight_decay=0)
        ids = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            target = teacher(ids, use_cache=False).logits.float().softmax(-1)
        losses = []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits = student(ids, use_cache=False).logits.float()
            loss = torch.nn.functional.kl_div(logits.log_softmax(-1), target, reduction="sum") / ids.numel()
            loss.backward()
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for _, p in named))
            optimizer.step()
            losses.append(float(loss.detach()))
        self.assertTrue(all(torch.equal(p, kept[n]) for n, p in student.named_parameters() if n in kept))
        with torch.no_grad():
            expected = student(ids, use_cache=False).logits
        exported = copy.deepcopy(student)
        self.assertEqual(replace(exported, packed=True), count)
        with torch.no_grad():
            actual = exported(ids, use_cache=False).logits
        self.assertTrue(torch.equal(expected, actual))
        print(json.dumps({"native_layers": 4, "selected_matrices": count, "actual_updates": 2,
                          "full_vocab_KL_losses": losses, "packed_reload_logits_bitwise_equal": True,
                          "retained_state_path_unchanged": True, "27B_training_completed": False}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
