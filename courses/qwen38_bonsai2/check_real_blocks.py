"""Real-weight layer0/3 controls; these do not train the full27B model."""
import argparse
import copy
import gc
import json
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
from control import sha, tensor_sha
from offload import backend
from policy import quantized_matrix
from ternary import RotatedTernary, PackedTernary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--source-proof", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--optimizer", choices=("adamw", "sgd"), required=True)
    p.add_argument("--lr", required=True, type=float)
    p.add_argument("--steps", type=int, default=2, choices=range(1, 9))
    a = p.parse_args()
    if a.output.exists() or not 0 < a.lr <= 1e-3:
        raise FileExistsError("Preserve prior block-control evidence")
    if sha(a.index) != "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df":
        raise ValueError("Expected pinned original27B source index")
    if sha(a.model / "config.json") != "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab":
        raise ValueError("Expected pinned original27B config")
    proof = json.loads(a.source_proof.read_text())
    if proof["revision"] != "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0":
        raise ValueError("Wrong original revision")
    index = json.loads(a.index.read_text())["weight_map"]
    prefixes = [f"model.language_model.layers.{i}." for i in (0, 3)]
    consumed = sorted({index[n] for n in index if any(n.startswith(pre) for pre in prefixes)} |
                      {index["model.language_model.embed_tokens.weight"]})
    known = {r["name"]: r for r in proof["completed_files"]}
    for name in consumed:
        if name not in known or sha(a.model / name) != known[name]["SHA256"]:
            raise ValueError("Required source shard is incomplete or changed")
    backend()
    torch.set_num_threads(4)
    torch.manual_seed(381027)
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose one admitted idle GPU")
    cfg = AutoConfig.from_pretrained(a.model, local_files_only=True).text_config
    cfg._attn_implementation = "sdpa"
    position = torch.arange(128, device="cuda")[None]
    with safe_open(str(a.model / index["model.language_model.embed_tokens.weight"]), framework="pt") as f:
        # Synthetic index fixture using real embeddings; not a capability test.
        hidden = torch.nn.functional.embedding(torch.arange(128)[None], f.get_tensor("model.language_model.embed_tokens.weight")).cuda()
    rope = qwen.Qwen3_5TextRotaryEmbedding(cfg, device="cuda")
    results = []
    for layer_index in (0, 3):
        prefix = f"model.language_model.layers.{layer_index}."
        with torch.device("meta"):
            teacher = qwen.Qwen3_5DecoderLayer(cfg, layer_index)
        values = {}
        for name in index:
            if name.startswith(prefix):
                with safe_open(str(a.model / index[name]), framework="pt") as f:
                    values[name[len(prefix):]] = f.get_tensor(name).cuda()
        teacher.load_state_dict(values, assign=True, strict=True)
        teacher.requires_grad_(False).eval()
        student = copy.deepcopy(teacher)
        selected = []
        for name, module in list(student.named_modules()):
            if quantized_matrix(prefix + name + ".weight"):
                parent, _, leaf = name.rpartition(".")
                setattr(student.get_submodule(parent) if parent else student, leaf, RotatedTernary(module.weight))
                selected.append(name)
        named = [(n, p) for n, p in student.named_parameters() if p.requires_grad]
        if len(named) != (6 if layer_index == 0 else 7):
            raise ValueError("Unexpected real-block quantization scope")
        retained = {n: tensor_sha(p) for n, p in student.named_parameters() if not p.requires_grad}
        def forward(model):
            return model(hidden, position_embeddings=rope(hidden, position), position_ids=position,
                         past_key_values=None, attention_mask=None)
        with torch.no_grad():
            target = forward(teacher)
        optimizer_type = torch.optim.AdamW if a.optimizer == "adamw" else torch.optim.SGD
        optimizer = optimizer_type([p for _, p in named], lr=a.lr, weight_decay=0, foreach=False)
        losses = []
        norms = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(a.steps):
            optimizer.zero_grad(set_to_none=True)
            output = forward(student)
            loss = (output.float() - target.float()).square().mean() / target.float().square().mean().clamp_min(1e-12)
            loss.backward()
            if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for _, p in named):
                raise ValueError("Real-block master gradient failed")
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in named], 1, foreach=False)
            if not bool(torch.isfinite(norm)):
                raise ValueError("Non-finite real-block norm")
            optimizer.step()
            losses.append(float(loss.detach()))
            norms.append(float(norm))
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            expected = forward(student)
            final_loss = float((expected.float() - target.float()).square().mean() /
                               target.float().square().mean().clamp_min(1e-12))
        if retained != {n: tensor_sha(p) for n, p in student.named_parameters() if n in retained}:
            raise ValueError("A retained state-path parameter changed")
        for name, module in list(student.named_modules()):
            if isinstance(module, RotatedTernary):
                parent, _, leaf = name.rpartition(".")
                setattr(student.get_submodule(parent) if parent else student, leaf, PackedTernary(module.packed()).cuda())
        with torch.no_grad():
            actual = forward(student)
        if not torch.equal(expected, actual):
            raise ValueError("Real-block packed reconstruction differs")
        result = {"layer": layer_index, "layer_type": cfg.layer_types[layer_index],
                  "selected_matrices": len(named), "trainable_elements": sum(p.numel() for _, p in named),
                  "updates": a.steps, "all_selected_gradients_finite": True, "retained_parameters_unchanged": True,
                  "normalized_block_output_MSE_losses": losses, "packed_output_bitwise_equal": True,
                  "post_final_update_normalized_block_output_MSE": final_loss,
                  "pre_clip_gradient_norms": norms,
                  "peak_GPU_allocated_bytes": torch.cuda.max_memory_allocated()}
        results.append(result)
        print(json.dumps(result), flush=True)
        del teacher, student, values, optimizer, named, output, target, loss, expected, actual, module
        gc.collect()
        torch.cuda.empty_cache()
    record = {"passed": True, "source_revision": proof["revision"],
              "optimizer": a.optimizer, "learning_rate": a.lr, "steps_per_block": a.steps,
              "consumed_original_shard_SHA256": {n: known[n]["SHA256"] for n in consumed}, "results": results,
              "full27_model_loaded": False, "full27_recovery_training_completed": False,
              "quality_accepted": False, "scope": "two individual real-weight blocks, synthetic embedding-index input, block-output MSE"}
    a.output.write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
