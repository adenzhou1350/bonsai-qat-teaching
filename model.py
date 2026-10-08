"""Text-only Qwen3.5 MoE, partitioned by complete layers across CUDA GPUs."""
import gc,json
from pathlib import Path
import torch
from safetensors import safe_open
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen
from quantization import QuantizedExperts

def backend():
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    qwen.FusedRMSNormGated = None
    qwen.causal_conv1d_fn = None
    qwen.chunk_gated_delta_rule = None
    qwen.fused_recurrent_gated_delta_rule = None

class TextModel(torch.nn.Module):
    def __init__(self, path, devices, quantized):
        super().__init__(); backend()
        path = Path(path); index = json.loads((path / 'model.safetensors.index.json').read_text())['weight_map']
        cfg = AutoConfig.from_pretrained(path, local_files_only=True).text_config
        assert cfg.model_type in ('qwen3_5_moe', 'qwen3_5_moe_text'), 'Only Qwen3.5 MoE is supported'
        cfg._attn_implementation = 'sdpa'; cfg._experts_implementation = 'eager'
        self.devices = [torch.device('cuda', d) for d in devices]
        self.slots = [i * len(devices) // cfg.num_hidden_layers for i in range(cfg.num_hidden_layers)]
        # FP32 master + gradient + two Adam moments = 16 bytes per expert value.
        elements = [0] * len(devices)
        for name, shard in index.items():
            if '.mlp.experts.' in name:
                layer = int(name.split('.layers.')[1].split('.')[0])
                with safe_open(str(path / shard), framework='pt') as f:
                    shape = f.get_slice(name).get_shape()
                elements[self.slots[layer]] += __import__('math').prod(shape)
        for slot, device in enumerate(self.devices):
            floor = elements[slot] * (16 if quantized else 2)
            free, _ = torch.cuda.mem_get_info(device)
            if floor + 8 * 2**30 > free:
                raise RuntimeError(f'{device}: expert state alone needs {floor / 2**30:.1f} GiB; '
                                   f'free {free / 2**30:.1f} GiB. CPU offload is not implemented.')
        def load(name, device):
            with safe_open(str(path / index[name]), framework='pt') as f:
                return f.get_tensor(name).to(device)
        self.layers = torch.nn.ModuleList()
        for i in range(cfg.num_hidden_layers):
            prefix = f'model.language_model.layers.{i}.'; device = self.devices[self.slots[i]]
            with torch.device('meta'): layer = qwen.Qwen3_5MoeDecoderLayer(cfg, i)
            values = {n[len(prefix):]: load(n, device) for n in index if n.startswith(prefix)}
            layer.load_state_dict(values, strict=True, assign=True); layer.requires_grad_(False)
            if quantized: layer.mlp.experts = QuantizedExperts(layer.mlp.experts)
            self.layers.append(layer); del values; gc.collect()
            print(f'Loaded layer {i + 1}/{cfg.num_hidden_layers} on {device}', flush=True)
        self.embedding = torch.nn.Parameter(load('model.language_model.embed_tokens.weight', self.devices[0]), requires_grad=False)
        self.head = torch.nn.Parameter(load('lm_head.weight', self.devices[-1]), requires_grad=False)
        self.norm = qwen.Qwen3_5MoeRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(self.devices[-1])
        self.norm.load_state_dict({'weight': load('model.language_model.norm.weight', self.devices[-1])}, assign=True)
        self.norm.requires_grad_(False)
        self.ropes = torch.nn.ModuleList([qwen.Qwen3_5MoeTextRotaryEmbedding(cfg, device=d) for d in self.devices])
        self.eval()
    def forward(self, ids, recompute=False):
        x = torch.nn.functional.embedding(ids.to(self.devices[0]), self.embedding)
        for i, layer in enumerate(self.layers):
            slot = self.slots[i]; x = x.to(self.devices[slot])
            def call(z, layer=layer, slot=slot):
                position = torch.arange(z.shape[1], device=z.device)[None]
                return layer(z, position_embeddings=self.ropes[slot](z, position),
                             position_ids=position, past_key_values=None)
            x = checkpoint(call, x, use_reentrant=False) if recompute else call(x)
        return self.norm(x)
