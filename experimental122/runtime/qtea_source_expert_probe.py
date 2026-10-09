"""Inference dependency extracted from validated experimental runtime."""

import hashlib

import torch

def tensor_sha(value):
    flat=value.detach().contiguous().view(torch.uint8).reshape(-1);digest=hashlib.sha256()
    for i in range(0,len(flat),16*1024*1024):digest.update(flat[i:i+16*1024*1024].numpy().tobytes())
    return digest.hexdigest()
