import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from webui.utils import check_paths
from webui import preset_loader


DEFAULT_MODEL_ROOT = "/workspace/TurboDiffusion/checkpoints"


def _model_search_roots() -> List[Path]:
    env_value = os.environ.get("MODEL_PATHS", "")
    if env_value.strip():
        raw_roots = [p.strip() for p in env_value.split(",") if p.strip()]
    else:
        raw_roots = [DEFAULT_MODEL_ROOT]

    roots: List[Path] = []
    for root in raw_roots:
        path = Path(root).expanduser()
        if path not in roots:
            roots.append(path)
    return roots


def _normalize_relative(path: Path) -> Path:
    if path.parts and path.parts[0] == "checkpoints":
        return Path(*path.parts[1:])
    return path


def _resolve_checkpoint_path(path_str: str, roots: List[Path]) -> str:
    path = Path(path_str)
    candidates = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(Path.cwd() / path)
        normalized = _normalize_relative(path)
        for root in roots:
            candidates.append(root / normalized)
            candidates.append(root / path)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(candidates[0])


def _resolve_preset_paths(cfg: "EngineConfig") -> "EngineConfig":
    roots = _model_search_roots()
    return EngineConfig(
        name=cfg.name,
        dit_path=_resolve_checkpoint_path(cfg.dit_path, roots),
        vae_path=_resolve_checkpoint_path(cfg.vae_path, roots),
        text_encoder_path=_resolve_checkpoint_path(cfg.text_encoder_path, roots),
        model=cfg.model,
        resolution=cfg.resolution,
        aspect_ratio=cfg.aspect_ratio,
        quant_linear=cfg.quant_linear,
        default_norm=cfg.default_norm,
    )

@dataclass(frozen=True)
class EngineConfig:
    name: str
    dit_path: str
    vae_path: str
    text_encoder_path: str
    model: str = "Wan2.1-1.3B"
    resolution: str = "480p"
    aspect_ratio: str = "16:9"
    quant_linear: bool = True
    default_norm: bool = False

PRESETS = {
    "Wan2.1 T2V 1.3B 480p (quant)": EngineConfig(
        name="Wan2.1 T2V 1.3B 480p (quant)",
        dit_path="checkpoints/TurboWan2.1-T2V-1.3B-480P-quant.pth",
        vae_path="checkpoints/Wan2.1_VAE.pth",
        text_encoder_path="checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
        model="Wan2.1-1.3B",
        resolution="480p",
        aspect_ratio="16:9",
        quant_linear=True,
        default_norm=False,
    ),
    "Wan2.1 T2V 14B 720p (quant, 5090 recommended)": EngineConfig(
        name="Wan2.1 T2V 14B 720p (quant, 5090 recommended)",
        dit_path="checkpoints/TurboWan2.1-T2V-14B-720P-quant.pth",
        vae_path="checkpoints/Wan2.1_VAE.pth",
        text_encoder_path="checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
        model="Wan2.1-14B",
        resolution="720p",
        aspect_ratio="16:9",
        quant_linear=True,      # quant checkpoint 需要 --quant_linear（官方建议）
        default_norm=False,
    ),

    "Wan2.1 T2V 14B 720p (fp16, >40GB GPU)": EngineConfig(
        name="Wan2.1 T2V 14B 720p (fp16, >40GB GPU)",
        dit_path="checkpoints/TurboWan2.1-T2V-14B-720P.pth",
        vae_path="checkpoints/Wan2.1_VAE.pth",
        text_encoder_path="checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
        model="Wan2.1-14B",
        resolution="720p",
        aspect_ratio="16:9",
        quant_linear=False,     # 非 quant checkpoint 不要开 quant_linear
        default_norm=False,
    ),

}


def load_presets() -> Dict[str, EngineConfig]:
    """Return merged presets including discovered checkpoints."""
    return preset_loader.discover_presets(PRESETS, EngineConfig)


def get_preset(name: str) -> EngineConfig:
    """Return preset config by name, resolved against MODEL_PATHS search roots."""
    cfg = PRESETS[name]
    return _resolve_preset_paths(cfg)


def available_preset_names() -> List[str]:
    """Return preset names whose checkpoint files all exist within search roots."""
    available = []
    for name, cfg in PRESETS.items():
        resolved = _resolve_preset_paths(cfg)
        if not check_paths(resolved):
            available.append(name)
    return available


def available_presets() -> Dict[str, EngineConfig]:
    """Return presets that have all required checkpoint files present."""
    return {name: get_preset(name) for name in available_preset_names()}


def discoverable_preset_names() -> List[str]:
    """Return preset names; prefer discovered ones, otherwise fallback to all."""
    discovered = available_preset_names()
    if discovered:
        return discovered
    return list(PRESETS.keys())
