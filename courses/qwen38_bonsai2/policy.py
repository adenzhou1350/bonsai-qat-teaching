"""Text-only scope follows the published Bonsai 2 matrix/retention boundary."""


def quantized_matrix(name):
    if name in ("model.language_model.embed_tokens.weight", "lm_head.weight"):
        return True
    if not name.startswith("model.language_model.layers."):
        return False
    return name.endswith(tuple("." + suffix + ".weight" for suffix in (
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "self_attn.q_proj",
        "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj")))


def text_tensor(name):
    return name == "lm_head.weight" or name.startswith("model.language_model.")
