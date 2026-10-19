"""Full-size TWO-UPDATE infrastructure control, not Bonsai quality recovery training.

CPU FP32 masters/gradients and plain SGD avoid Adam's full-size moments. This is
an explicit control-only optimizer choice, not a recovered official training recipe.
The packed correctness runtime expands individual matrices; it is not a fused kernel.
"""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer
from offload import load_model, StagedTernary, backend
from ternary import PackedTernary


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data := stream.read(8 * 1024 * 1024):
            h.update(data)
    return h.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def memory_admission(elements):
    cgroup = Path("/sys/fs/cgroup")
    maximum = (cgroup / "memory.max").read_text().strip()
    stat = dict(line.split() for line in (cgroup / "memory.stat").read_text().splitlines())
    # Cached source files are reclaimable; account for anonymous/shared memory.
    existing = int(stat["anon"]) + int(stat.get("shmem", 0))
    needed = elements * 8 + 36 * 2**30
    if maximum == "max" or existing + needed >= int(maximum):
        raise RuntimeError("A known cgroup budget must admit masters, gradients, scratch and reserves")
    return {"cgroup_limit_bytes": int(maximum), "existing_anon_and_shmem_bytes": existing,
            "planned_master_gradient_scratch_and_reserve_bytes": needed,
            "optimizer": "SGD_without_momentum_control_only"}


def packed_reload(directory, device):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    backend()
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if sha(directory / "config.json") != manifest["config_SHA256"]:
        raise ValueError("Packed config hash mismatch")
    cfg = Qwen3_5TextConfig(**json.loads((directory / "config.json").read_text()))
    cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(cfg)
    for row in manifest["matrices"]:
        path = directory / row["file"]
        if sha(path) != row["SHA256"]:
            raise ValueError("Packed matrix hash mismatch")
        value = torch.load(path, map_location="cpu", weights_only=True)
        parent, _, leaf = row["module"].rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, leaf, PackedTernary(value).to(device))
    if sha(directory / "retained.pt") != manifest["retained_SHA256"]:
        raise ValueError("Retained tensor hash mismatch")
    retained = torch.load(directory / "retained.pt", map_location="cpu", weights_only=True)
    for name, value in retained.items():
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, torch.nn.Parameter(value.to(device), requires_grad=False))
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(cfg, device=device)
    if any(p.is_meta for p in model.parameters()):
        raise ValueError("Packed reload left meta parameters")
    return model.eval(), manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--source-manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--steps", type=int, default=2, choices=[1, 2])
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    a = p.parse_args()
    if not 16 <= a.seq_len <= 128 or not 0 < a.lr <= 1e-3:
        raise ValueError("This is a bounded small-token control")
    a.output.mkdir(parents=True, exist_ok=False)
    source = json.loads(a.source_manifest.read_text())
    if source.get("revision") != "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" or not source.get("passed"):
        raise ValueError("Pinned, verified source download required")
    files = source["completed_files"]
    if len([r for r in files if r["name"].endswith(".safetensors")]) != 18:
        raise ValueError("Incomplete original source")
    for row in files:
        if sha(a.model / row["name"]) != row["SHA256"]:
            raise ValueError("Original model file changed")
    torch.set_num_threads(8)
    torch.manual_seed(381027)
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose only the freshly admitted idle GPU")
    device = torch.device("cuda", 0)
    report = {"status": "starting_full_size_two_update_control", "actual_optimizer_updates": 0,
              "full_recovery_training_complete": False, "official_training_recipe_reproduced": False,
              "quality_accepted": False, "vision_or_MTP_trained": False,
              "source_revision": source["revision"], "source_manifest_SHA256": sha(a.source_manifest),
              "memory_admission": memory_admission(26869760000), "events": [],
              "CPU_FP32_masters": True, "CPU_weight_offload_during_training": True,
              "fused_deployment_kernel": False, "resumable_master_checkpoint_saved": False}
    started = time.monotonic()
    def save(stage):
        report.update(status=stage, seconds=time.monotonic() - started)
        tmp = a.output / "progress.tmp"
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(a.output / "progress.json")
    save("loading_original_BF16_teacher_with_CPU_weight_offload")
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    # Authored mechanism fixtures, kept separate from capability evaluation datasets.
    texts = [
        "A matrix maps input vectors to output vectors. Quantization approximates its weights using a small set of values and a scale. A normalized orthogonal transform changes coordinates while preserving the original linear map when both the weights and inputs are transformed consistently. " * 8,
        "A program can preserve its training state by saving the parameters, optimizer state, random state, and input identity. Restoring those values allows the next optimization step to be checked independently. A compact exported model and a resumable training checkpoint serve different purposes. " * 8,
        "Validation examples measure behavior without contributing gradients. A lower distillation loss on training examples does not establish accuracy on new questions. Quality evaluation needs separate inputs, consistent generation settings, and an original-model reference. " * 8,
    ]
    rows = [torch.tensor(tokenizer.encode(t, add_special_tokens=False)[:a.seq_len], dtype=torch.long)[None].to(device) for t in texts]
    if not all(x.numel() == a.seq_len for x in rows):
        raise ValueError("Control fixture is too short")
    report["fixture_input_IDs"] = [x.cpu().tolist() for x in rows]
    report["fixture_scope"] = "two authored TRAIN texts; one separate authored VAL text; not a quality panel"
    teacher, _ = load_model(a.model, device, quantized=False)
    with torch.no_grad():
        targets = [teacher.model(ids, use_cache=False).last_hidden_state.cpu() for ids in rows]
    teacher_head = teacher.lm_head
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    save("loading_all402_rotated_FP32_student_masters_on_CPU")
    student, cfg = load_model(a.model, device, quantized=True)
    named = [(n, p) for n, p in student.named_parameters() if p.requires_grad]
    assert len(named) == 402 and sum(p.numel() for _, p in named) == 26869760000
    kept = {n: tensor_sha(p) for n, p in student.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.SGD([p for _, p in named], lr=a.lr, momentum=0, foreach=False)
    student.train()
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    def objective(index):
        prediction = student.model(rows[index], use_cache=False).last_hidden_state
        target = targets[index].to(device)
        def part(x, y):
            logits = student.lm_head(x).float()
            with torch.no_grad():
                probabilities = teacher_head(y).float().softmax(-1)
            return torch.nn.functional.kl_div(logits.log_softmax(-1), probabilities, reduction="sum")
        kl = prediction.new_zeros((), dtype=torch.float32)
        for low in range(0, a.seq_len, 8):
            args = prediction[:, low:low+8], target[:, low:low+8]
            kl = kl + (checkpoint(part, *args, use_reentrant=False) if torch.is_grad_enabled() else part(*args))
        kl = kl / a.seq_len
        mse = (prediction.float() - target.float()).square().mean() / target.float().square().mean().clamp_min(1e-12)
        return kl + .1*mse, kl, mse
    with torch.no_grad():
        report["validation_before"] = [float(x) for x in objective(2)]
    save("running_full64_layers_all402_matrices_control_updates")
    for step in range(a.steps):
        optimizer.zero_grad(set_to_none=True)
        loss, kl, mse = objective(step)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Non-finite control loss")
        loss.backward()
        for _, param in named:
            if param.grad is None or param.grad.device.type != "cpu":
                raise ValueError("Missing CPU master gradient")
            for low in range(0, param.shape[0], 128):
                if not bool(torch.isfinite(param.grad[low:low+128]).all()):
                    raise ValueError("Non-finite CPU gradient")
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in named], 1., foreach=False)
        if not bool(torch.isfinite(norm)):
            raise ValueError("Non-finite gradient norm")
        optimizer.step()
        report["actual_optimizer_updates"] += 1
        report["events"].append({"step": step+1, "TRAIN_row": step, "loss": float(loss.detach()),
                                 "KL": float(kl.detach()), "state_MSE": float(mse.detach()),
                                 "finite_CPU_gradient_matrices": len(named), "gradient_norm": float(norm)})
        del loss, kl, mse
        optimizer.zero_grad(set_to_none=True)
        save("completed_control_update_" + str(step+1))
    with torch.no_grad():
        report["validation_after"] = [float(x) for x in objective(2)]
        reference = student.lm_head(student.model(rows[2], use_cache=False).last_hidden_state[:, -1:]).cpu()
    assert kept == {n: tensor_sha(p) for n, p in student.named_parameters() if n in kept}
    report["retained_tensor_hashes_unchanged"] = True
    packed = a.output / "packed"
    packed.mkdir()
    (packed / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    matrices = []
    save("exporting_same_GPU_ternary_grids_and_FP16_scales")
    for name, module in student.named_modules():
        if not isinstance(module, StagedTernary):
            continue
        value = module.packed(device=device)
        path = packed / f"{len(matrices):03d}.pt"
        torch.save(value, path)
        matrices.append({"module": name, "file": path.name, "SHA256": sha(path)})
        del value
    torch.save({n: p.detach().cpu() for n, p in student.named_parameters() if n in kept}, packed / "retained.pt")
    manifest = {"format": "teaching-qwen38-rotated-ternary-control-v1", "matrices": matrices,
                "config_SHA256": sha(packed / "config.json"),
                "retained_SHA256": sha(packed / "retained.pt"), "actual_optimizer_updates": a.steps,
                "source_revision": source["revision"], "quality_accepted": False,
                "official_Bonsai_GGUF_compatible": False, "fused_kernel": False,
                "retained_state_hashes": kept, "control_only": True}
    (packed / "manifest.json").write_text(json.dumps(manifest, indent=2))
    del optimizer, named, student, teacher_head
    gc.collect()
    torch.cuda.empty_cache()
    save("independent_model_reconstruction_from_packed_files")
    reloaded, _ = packed_reload(packed, device)
    with torch.no_grad():
        actual = reloaded.lm_head(reloaded.model(rows[2], use_cache=False).last_hidden_state[:, -1:]).cpu()
    report["reloaded_last_position_logits_bitwise_equal"] = torch.equal(reference, actual)
    if not report["reloaded_last_position_logits_bitwise_equal"]:
        raise ValueError("Packed reload differs from the trained control")
    report.update(passed=True, selected_matrices=len(matrices),
                  packed_manifest_SHA256=sha(packed / "manifest.json"),
                  packed_file_bytes=sum(p.stat().st_size for p in packed.iterdir()),
                  peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
                  limitation="Only two short TRAIN updates; no recovered capability or production inference claim")
    save("full_size_two_update_control_export_reload_completed")
    (a.output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
