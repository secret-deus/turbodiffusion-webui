import os
import torch

def has_spargeattn():
    try:
        import spargeattn  # noqa
        return True
    except Exception:
        return False

def gpu_info():
    if not torch.cuda.is_available():
        return {"cuda": False, "msg": "CUDA not available"}
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    free, total = torch.cuda.mem_get_info()
    return {
        "cuda": True,
        "device": idx,
        "name": name,
        "free_gb": round(free / 1024**3, 2),
        "total_gb": round(total / 1024**3, 2),
        "max_alloc_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    }

def check_paths(cfg):
    missing = []
    paths = [cfg.dit_path, cfg.vae_path, cfg.text_encoder_path]
    dit_path_high = getattr(cfg, "dit_path_high", None)
    if dit_path_high:
        paths.append(dit_path_high)

    for p in paths:
        if not os.path.exists(p):
            missing.append(p)
    return missing
