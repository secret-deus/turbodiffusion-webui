import json
import re
from pathlib import Path
from typing import Dict, Tuple


RESOLUTION_MAP: Dict[str, Tuple[str, str]] = {
    "480P": ("480p", "16:9"),
    "720P": ("720p", "16:9"),
}

# 模型大小映射 - 支持 T2V 和 I2V
SIZE_MODEL_MAP = {
    "1.3B": "Wan2.1-1.3B",
    "14B": "Wan2.1-14B",
    "A14B": "Wan2.2-A14B",  # Wan2.2 I2V A14B 模型
}

# 模型版本映射
VERSION_MAP = {
    "2.1": "Wan2.1",
    "2.2": "Wan2.2",
}


def _tokenize(name: str):
    # normalize underscores/dashes and case for matching
    return re.split(r"[_\-]+", name.upper())


def _load_sidecar(ckpt_path: Path) -> Dict[str, object]:
    """Load optional metadata sidecar next to the checkpoint.

    We prefer a same-name ``.json`` file, and fall back to ``model.json``
    sitting alongside the checkpoint. Invalid JSON is ignored.
    """

    candidates = [ckpt_path.with_suffix(".json"), ckpt_path.parent / "model.json"]
    for meta_path in candidates:
        if not meta_path.exists():
            continue
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            # ignore malformed metadata but continue to other candidates
            continue
    return {}


def infer_from_checkpoint(ckpt_path: Path) -> Tuple[Dict[str, object], str]:
    """Infer engine defaults from a checkpoint filename and optional sidecar.

    Returns a tuple of (merged_fields, info_string) where ``merged_fields``
    contains values for ``model``, ``resolution``, ``aspect_ratio``, and
    ``quant_linear`` when they can be inferred. Sidecar metadata overrides any
    inference.
    """

    tokens = _tokenize(ckpt_path.stem)
    inferred: Dict[str, object] = {}

    # quant checkpoints usually include "quant" in the filename
    if "QUANT" in tokens:
        inferred["quant_linear"] = True
    else:
        inferred["quant_linear"] = False

    # resolution/aspect defaults from common tokens
    for token in tokens:
        if token in RESOLUTION_MAP:
            inferred["resolution"], inferred["aspect_ratio"] = RESOLUTION_MAP[token]
            break

    # model size tokens like 1.3B / 14B / A14B
    for token in tokens:
        if token in SIZE_MODEL_MAP:
            inferred["model"] = SIZE_MODEL_MAP[token]
            break

    # version detection (Wan2.1, Wan2.2)
    # 如果在 SIZE_MODEL_MAP 中已经匹配到带版本的模型名，则使用它
    # 否则尝试从文件名推断版本
    if "model" not in inferred:
        for token in tokens:
            if token in VERSION_MAP:
                # 如果有版本但没有模型，使用默认模型
                version = VERSION_MAP[token]
                inferred["model"] = f"{version}-14B"  # 默认 14B
                break

    overrides = _load_sidecar(ckpt_path)

    sources: Dict[str, str] = {}

    def pick(key: str, default=None):
        if key in overrides:
            sources[key] = "metadata"
            return overrides[key]
        if key in inferred:
            sources[key] = "filename"
            return inferred[key]
        sources[key] = "default"
        return default

    merged = {
        "model": pick("model"),
        "resolution": pick("resolution"),
        "aspect_ratio": pick("aspect_ratio"),
        "quant_linear": bool(pick("quant_linear", False)),
        "default_norm": pick("default_norm"),
        "name": overrides.get("name"),
        "vae_path": overrides.get("vae_path"),
        "text_encoder_path": overrides.get("text_encoder_path"),
    }

    info_bits = []
    for key in ("model", "resolution", "aspect_ratio", "quant_linear", "default_norm"):
        if merged.get(key) is None:
            continue
        info_bits.append(f"{key}={merged[key]} ({sources.get(key,'default')})")
    if overrides:
        applied = ", ".join(sorted(overrides.keys()))
        info_bits.append(f"overrides: {applied}")

    info = " | ".join(info_bits)
    return merged, info


def discover_checkpoints(checkpoints_dir: Path) -> Dict[str, Path]:
    """Return a mapping of discovered checkpoint names to paths.

    We consider Wan2.x Turbo checkpoints (T2V and I2V) to avoid picking up
    unrelated weights that may live in the same folder.
    """

    discovered: Dict[str, Path] = {}
    if not checkpoints_dir.exists():
        return discovered

    # 发现 T2V 模型（Wan2.1）
    for path in checkpoints_dir.glob("*Wan2.1*T2V*.pth"):
        discovered[path.stem] = path

    # 发现 I2V 模型（Wan2.2）
    for path in checkpoints_dir.glob("*Wan2.2*I2V*.pth"):
        discovered[path.stem] = path

    return discovered
