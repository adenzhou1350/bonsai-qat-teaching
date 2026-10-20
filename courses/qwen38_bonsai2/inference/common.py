"""Portable identity checks; importing this module does not import Torch."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COURSE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(COURSE))

MODES = ("reference", "decode", "decode_rotate")
PROMPTS_SHA256 = "62a7f9bc8a2627013b6553d97628c9018df2f07e983a0648e6250dc421548ba3"
PERFORMANCE_SCOPE = (
    "Batch 1 greedy generation, dynamic KV cache, two-token untimed warmup. "
    "First-token time includes prefill. Decode time includes model forward, "
    "argmax and CUDA synchronization; excludes loading, tokenization, CPU "
    "logit copies/hashing, finite checks, token.item, EOS handling and output. "
    "This is warm compute timing, not end-to-end service throughput."
)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def verify_sources():
    files = json.loads((HERE / "source-files.json").read_text(encoding="utf-8"))
    for name, expected in files.items():
        path = (COURSE / name).resolve()
        if not path.is_relative_to(COURSE) or sha(path) != expected:
            raise ValueError("Source identity mismatch: " + name)
    return files


def check_imports():
    files = verify_sources()
    names = ("kernels", "runtime", "rotation", "ternary", "control", "offload", "policy")
    for name in names:
        path = Path(sys.modules[name].__file__).resolve()
        expected = HERE / (name + ".py") if name in ("kernels", "runtime") else COURSE / (name + ".py")
        if path != expected or sha(path) != files[path.relative_to(COURSE).as_posix()]:
            raise ValueError("Unexpected imported module: " + name)


def verify_artifacts(packed, tokenizer=None):
    verify_sources()
    from infer_prompt import verify
    manifest, identity = verify(packed)
    if tokenizer is not None:
        root = tokenizer.resolve()
        files = json.loads((HERE / "tokenizer-files.json").read_text(encoding="utf-8"))
        for name, expected in files.items():
            path = (root / name).resolve()
            if Path(name).name != name or not path.is_relative_to(root) or sha(path) != expected:
                raise ValueError("Tokenizer identity mismatch: " + name)
    return manifest, identity


def load(packed, tokenizer_path):
    import torch
    from transformers import AutoTokenizer
    from control import packed_reload
    from runtime import install
    check_imports()
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one idle, reserved CUDA GPU")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.9)
    model, manifest = packed_reload(packed, torch.device("cuda", 0))
    model.requires_grad_(False)
    if install(model) != 402:
        raise ValueError("Expected all 402 recorded packed matrices")
    tensors = [*model.parameters(), *model.buffers()]
    if not tensors or not all(t.is_cuda for t in tensors):
        raise ValueError("All model tensors must remain on CUDA")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    raw = model.generation_config.eos_token_id
    eos = set(raw if isinstance(raw, list) else [raw]) | {tokenizer.eos_token_id}
    return model, tokenizer, eos, manifest


def encode(tokenizer, prompt):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not 1 <= len(tokens) <= 128:
        raise ValueError("Use 1..128 input tokens including the chat template")
    return tokens


def generate(model, eos, tokens, count, collect=True):
    import time
    import torch
    ids = torch.tensor(tokens, device="cuda")[None]
    cache, generated, times, hashes = None, [], [], []
    with torch.inference_mode():
        for _ in range(count):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = model(ids, use_cache=True, past_key_values=cache)
            cache = out.past_key_values
            logits = out.logits[:, -1].contiguous()
            token = logits.argmax(-1)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            if collect:
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("Nonfinite logits")
                hashes.append(hashlib.sha256(
                    logits.cpu().contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest())
            generated.append(int(token.item()))
            ids = token[:, None]
            if generated[-1] in eos:
                break
    return generated, times, hashes


def save_new(path, value):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
