import os
import subprocess
import torch

def gpu_info():
    info = {}
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        info["name"] = torch.cuda.get_device_name(i)
        info["capability"] = torch.cuda.get_device_capability(i)
        info["total_gb"] = round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2)
    else:
        info["name"] = "CUDA not available"
    return info

def has_spargeattn():
    try:
        import spargeattn  # noqa: F401
        return True
    except Exception:
        return False

def check_paths(cfg):
    missing = []
    for p in [cfg.dit_path, cfg.vae_path, cfg.text_encoder_path]:
        if not os.path.exists(p):
            missing.append(p)
    return missing