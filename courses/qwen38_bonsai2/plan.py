"""Inspect shapes and standard Adam state cost on meta; never load model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from policy import quantized_matrix, text_tensor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Preserve prior evidence; choose a new output")
    cfgdata = json.loads(a.config.read_text(encoding="utf-8"))
    cfg = AutoConfig.for_model(cfgdata.pop("model_type"), **cfgdata)
    if cfg.text_config.hidden_size != 5120 or cfg.text_config.num_hidden_layers != 64:
        raise ValueError("This audit expects the pinned Qwen3.8-27B architecture")
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen3_5ForConditionalGeneration(cfg)
    shapes = {n: list(p.shape) for n, p in model.named_parameters()}
    index = json.loads(a.index.read_text(encoding="utf-8"))["weight_map"]
    selected = {n: shape for n, shape in shapes.items() if quantized_matrix(n)}
    if len(selected) != 402 or not set(selected) <= set(index):
        raise ValueError("Unexpected text matrix coverage or missing source weights")
    if not all(len(s) == 2 and s[1] % 1024 == 0 and s[1] % 128 == 0 for s in selected.values()):
        raise ValueError("Unsupported rotation/group shape")
    retained = {n: s for n, s in shapes.items() if text_tensor(n) and not quantized_matrix(n)}
    count = sum(__import__("math").prod(s) for s in selected.values())
    kept = sum(__import__("math").prod(s) for s in retained.values())
    report = {"status": "meta_shape_and_resource_audit_only", "selected_matrices": selected,
              "retained_text_tensors": retained, "selected_matrix_count": len(selected),
              "trainable_master_elements": count, "retained_text_elements": kept,
              "FP32_master_gradient_Adam_bytes": count * 16,
              "FP32_master_gradient_Adam_GiB": count * 16 / 2**30,
              "checkpoint_master_Adam_bytes_excludes_metadata": count * 12,
              "source_config_SHA256": hashlib.sha256(a.config.read_bytes()).hexdigest(),
              "source_index_SHA256": hashlib.sha256(a.index.read_bytes()).hexdigest(),
              "model_weights_loaded": False, "optimizer_updates": 0,
              "CUDA_initialized": torch.cuda.is_initialized(),
              "excluded_components": ["vision tower", "MTP"],
              "packed_reference_is_fused_inference_kernel": False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("selected_matrices", "retained_text_tensors")}))


if __name__ == "__main__":
    main()
