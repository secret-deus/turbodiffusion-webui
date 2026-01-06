from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from webui.discovery import discover_checkpoints, infer_from_checkpoint

from webui.utils import check_paths

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
    info: str = ""

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


def _build_discovered_presets() -> Dict[str, EngineConfig]:
    """Create EngineConfig entries from locally discovered checkpoints."""

    discovered: Dict[str, EngineConfig] = {}
    checkpoints_dir = Path("checkpoints")
    vae_default = checkpoints_dir / "Wan2.1_VAE.pth"
    text_encoder_default = checkpoints_dir / "models_t5_umt5-xxl-enc-bf16.pth"

    for stem, path in discover_checkpoints(checkpoints_dir).items():
        fields, info = infer_from_checkpoint(path)

        cfg = EngineConfig(
            name=str(fields.get("name") or f"Auto: {stem}"),
            dit_path=str(path),
            vae_path=str(fields.get("vae_path") or vae_default),
            text_encoder_path=str(fields.get("text_encoder_path") or text_encoder_default),
            model=str(fields.get("model") or EngineConfig.model),
            resolution=str(fields.get("resolution") or EngineConfig.resolution),
            aspect_ratio=str(fields.get("aspect_ratio") or EngineConfig.aspect_ratio),
            quant_linear=bool(fields.get("quant_linear", EngineConfig.quant_linear)),
            default_norm=bool(fields.get("default_norm", EngineConfig.default_norm)),
            info=info,
        )
        discovered[cfg.name] = cfg
    return discovered


PRESETS.update(_build_discovered_presets())


def get_preset(name: str) -> EngineConfig:
    """Return preset config by name."""
    return PRESETS[name]


def available_preset_names() -> List[str]:
    """Return preset names whose checkpoint files all exist."""
    available = []
    for name, cfg in PRESETS.items():
        if not check_paths(cfg):
            available.append(name)
    return available


def available_presets() -> Dict[str, EngineConfig]:
    """Return presets that have all required checkpoint files present."""
    return {name: PRESETS[name] for name in available_preset_names()}


def preset_details(name: str) -> str:
    """Human readable summary for UI tooltips/status."""

    cfg = get_preset(name)
    parts = [
        f"Model: `{cfg.model}`",
        f"Resolution: {cfg.resolution}",
        f"Aspect ratio: {cfg.aspect_ratio}",
        f"quant_linear: {cfg.quant_linear}",
        f"default_norm: {cfg.default_norm}",
    ]
    if cfg.info:
        parts.append(cfg.info)
    return " | ".join(parts)
