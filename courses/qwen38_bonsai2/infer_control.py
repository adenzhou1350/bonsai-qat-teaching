"""Fresh-process inference of the two-update control, not official Bonsai GGUF."""
import argparse
import json
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from control import packed_reload, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packed", required=True, type=Path)
    p.add_argument("--tokenizer", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--max-new-tokens", type=int, default=32)
    a = p.parse_args()
    if a.output.exists() or not 1 <= a.max_new_tokens <= 64:
        raise ValueError("Use a new output path and a bounded generation")
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose one admitted CUDA GPU")
    torch.set_num_threads(4)
    model, manifest = packed_reload(a.packed, torch.device("cuda", 0))
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": "用一句话解释模型量化。"}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False), device="cuda")[None]
    if ids.numel() > 128:
        raise ValueError("Short control prompt only")
    torch.cuda.reset_peak_memory_stats()
    generated = []
    timings = []
    cache = None
    eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
              else [model.generation_config.eos_token_id])
    eos.add(tokenizer.eos_token_id)
    with torch.inference_mode():
        for _ in range(a.max_new_tokens):
            torch.cuda.synchronize()
            started = time.perf_counter()
            out = model(ids, use_cache=True, past_key_values=cache)
            cache = out.past_key_values
            token = out.logits[:, -1].argmax(-1)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - started)
            generated.append(int(token.item()))
            ids = token[:, None]
            if generated[-1] in eos:
                break
    stores = {v.untyped_storage().data_ptr(): v.untyped_storage().nbytes()
              for v in [*model.parameters(), *model.buffers()] if v.is_cuda}
    result = {"completed": True, "source_control_updates": manifest["actual_optimizer_updates"],
              "packed_manifest_SHA256": sha(a.packed / "manifest.json"), "generated_token_IDs": generated,
              "text": tokenizer.decode(generated, skip_special_tokens=True),
              "request_seconds_excludes_load_tokenization": sum(timings), "first_token_seconds": timings[0],
              "decode_after_first_tokens_per_second": (len(timings)-1) / sum(timings[1:]) if len(timings)>1 else None,
              "model_unique_CUDA_storage_bytes": sum(stores.values()),
              "peak_GPU_allocated_bytes": torch.cuda.max_memory_allocated(),
              "CPU_weight_offload_during_inference": False, "fused_kernel": False,
              "quality_accepted": False, "full_recovery_training_completed": False,
              "limitation": "One short prompt, packed-weight expansion reference, no quality or sustained-throughput claim"}
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
