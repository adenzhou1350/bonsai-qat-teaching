"""CPU masters with differentiable CUDA copies for bounded experimental QAT controls."""
import gc
import json
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
from policy import quantized_matrix
from rotation import rotate
from ternary import RotatedTernary, STE, quantize, unpack_grid


class StagedTernary(RotatedTernary):
    def __init__(self, original, embedding=False, block=1024, group=128, seed=381027,
                 compute_dtype=torch.bfloat16):
        if original.device.type != "cpu":
            raise ValueError("Staged masters must start on CPU")
        super().__init__(original, embedding, block, group, seed)
        self.compute_dtype = compute_dtype

    def forward(self, x):
        master = self.master.to(x.device)
        code, scale = quantize(master, self.group)
        grid = unpack_grid(code, scale, self.group).to(self.compute_dtype)
        weight = STE.apply(master, grid)
        if self.embedding:
            return rotate(torch.nn.functional.embedding(x, weight), self.signs, self.block, inverse=True)
        return torch.nn.functional.linear(rotate(x, self.signs, self.block), weight)

    @torch.no_grad()
    def packed(self, device=None):
        from ternary import pack
        master = self.master.to(device) if device is not None else self.master
        code, scales = quantize(master, self.group)
        return {"format": "teaching-rotated-ternary-v1", "codes": pack(code).cpu(),
                "scales": scales.cpu(), "shape": list(self.master.shape),
                "signs": self.signs.cpu(), "block": self.block, "group": self.group,
                "embedding": self.embedding,
                "compute_dtype": str(self.compute_dtype).split(".")[-1]}


class StagedLinear(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.detach().clone(), requires_grad=False)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight.to(device=x.device, dtype=x.dtype))


class StagedEmbedding(StagedLinear):
    def forward(self, ids):
        return torch.nn.functional.embedding(ids.cpu(), self.weight).to(ids.device)


def source_name(name):
    return name if name == "lm_head.weight" else name.replace("model.", "model.language_model.", 1)


def backend():
    # Match teacher/student implementation; keep this separate from fused deployment.
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    qwen.FusedRMSNormGated = None
    qwen.causal_conv1d_fn = qwen.causal_conv1d_update = None
    qwen.chunk_gated_delta_rule = qwen.fused_recurrent_gated_delta_rule = None


def load_model(path, device, quantized=False, on_loaded=None):
    """Load text matrices one at a time. Vision and MTP are explicitly excluded."""
    backend()
    path = Path(path)
    cfg = AutoConfig.from_pretrained(path, local_files_only=True).text_config
    if cfg.hidden_size != 5120 or cfg.num_hidden_layers != 64 or cfg.model_type != "qwen3_5_text":
        raise ValueError("Expected Qwen3.8-27B dense text architecture")
    cfg._attn_implementation = "sdpa"
    cfg.use_cache = False
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    with torch.device("meta"):
        model = qwen.Qwen3_5ForCausalLM(cfg)
    model.requires_grad_(False)
    selected = []
    # Top-level modules change during replacement; retain the original names/shapes.
    names = [(n, tuple(p.shape)) for n, p in model.named_parameters()]
    for name, shape in names:
        key = source_name(name)
        with safe_open(str(path / index[key]), framework="pt", device="cpu") as stream:
            value = stream.get_tensor(key)
        if tuple(value.shape) != shape:
            raise ValueError(f"Source shape mismatch: {key}")
        module_name, _, parameter = name.rpartition(".")
        parent_name, _, leaf = module_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        if quantized_matrix(key):
            assert parameter == "weight"
            if quantized:
                replacement = StagedTernary(value, embedding=module_name.endswith("embed_tokens"))
            else:
                replacement = (StagedEmbedding if module_name.endswith("embed_tokens") else StagedLinear)(value)
            setattr(parent, leaf, replacement)
            selected.append(name)
        else:
            owner = model.get_submodule(module_name)
            setattr(owner, parameter, torch.nn.Parameter(value.to(device, copy=True), requires_grad=False))
        del value
        if on_loaded:
            on_loaded(name, len(selected))
    if len(selected) != 402 or any(p.is_meta for p in model.parameters()):
        raise ValueError("Incomplete dense text load")
    # Rotary buffers are nonpersistent and therefore absent from the source index.
    model.model.rotary_emb = qwen.Qwen3_5TextRotaryEmbedding(cfg, device=device)
    model.eval()
    gc.collect()
    return model, cfg
