from __future__ import annotations

import torch


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device, requested: str = "auto") -> torch.dtype:
    if requested != "auto":
        return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[requested]
    return torch.float16 if device.type == "cuda" else torch.float32


def generate_kwargs(model, num_beams: int = 1, max_new_tokens: int = 128) -> dict:
    kw = {"num_beams": num_beams, "max_new_tokens": max_new_tokens}
    # English-only checkpoints (*.en) reject language/task arguments.
    if getattr(model.generation_config, "is_multilingual", True):
        kw.update(language="en", task="transcribe")
    return kw
