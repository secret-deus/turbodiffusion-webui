import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from webui.discovery import discover_checkpoints, infer_from_checkpoint

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
        info=cfg.info,
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


def _discovery_dirs() -> List[Path]:
    """Directories to scan for additional checkpoints.

    Always includes the local ``./checkpoints`` folder (relative to CWD) for
    backwards compatibility, then adds any ``MODEL_PATHS`` roots.
    """

    candidates: List[Path] = [Path("checkpoints"), *_model_search_roots()]

    # Some users may point MODEL_PATHS at a repo root; scan ``root/checkpoints`` too.
    for root in list(candidates):
        if root.name != "checkpoints":
            candidates.append(root / "checkpoints")

    seen: set[str] = set()
    dirs: List[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            resolved = candidate
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        dirs.append(resolved)
    return dirs


def _build_discovered_presets() -> Dict[str, EngineConfig]:
    """Create EngineConfig entries from locally discovered checkpoints."""

    discovered: Dict[str, EngineConfig] = {}
    default_dir = Path("checkpoints")
    vae_default = default_dir / "Wan2.1_VAE.pth"
    text_encoder_default = default_dir / "models_t5_umt5-xxl-enc-bf16.pth"

    for checkpoints_dir in _discovery_dirs():
        for stem, path in discover_checkpoints(checkpoints_dir).items():
            fields, info = infer_from_checkpoint(path)

            name_base = str(fields.get("name") or f"Auto: {stem}")
            name = name_base
            idx = 1
            while name in discovered:
                idx += 1
                name = f"{name_base} ({idx})"

            cfg = EngineConfig(
                name=name,
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


def _is_auto_preset_name(name: str) -> bool:
    stripped = name.lstrip()
    return stripped.startswith("Auto:") or stripped.startswith("Auto |")


def _hide_duplicate_auto_presets(names: List[str]) -> List[str]:
    """Hide auto-discovered presets when an equivalent named preset exists.

    We treat presets as "equivalent" if they point to the same DiT checkpoint
    filename (regardless of directory). This keeps the UI list clean while
    still allowing Auto entries for checkpoints that don't have a curated name.
    """

    manual_dit_filenames = set()
    for preset_name in names:
        if _is_auto_preset_name(preset_name):
            continue
        cfg = get_preset(preset_name)
        manual_dit_filenames.add(Path(cfg.dit_path).name.casefold())

    filtered: List[str] = []
    for preset_name in names:
        if _is_auto_preset_name(preset_name):
            cfg = get_preset(preset_name)
            dit_filename = Path(cfg.dit_path).name.casefold()
            if dit_filename in manual_dit_filenames:
                continue
        filtered.append(preset_name)

    return filtered


def discoverable_preset_names() -> List[str]:
    """Return preset names; prefer discovered ones, otherwise fallback to all."""
    discovered = available_preset_names()
    if discovered:
        return _hide_duplicate_auto_presets(discovered)
    return _hide_duplicate_auto_presets(list(PRESETS.keys()))


def refresh_discovered_presets() -> None:
    """Rescan local checkpoints and merge into :data:`PRESETS`."""

    PRESETS.update(_build_discovered_presets())


def preset_details(name: str) -> str:
    """Human-friendly markdown details for a preset.

    Used by the Gradio UI to show resolved paths + inferred metadata.
    """

    cfg = get_preset(name)
    missing = check_paths(cfg)

    status = "✅" if not missing else "❌"
    lines = [f"### {status} {cfg.name}"]

    if cfg.info:
        lines.append(f"- info: {cfg.info}")

    lines.extend(
        [
            f"- model: `{cfg.model}`",
            f"- resolution: `{cfg.resolution}` | aspect: `{cfg.aspect_ratio}`",
            f"- quant_linear: `{cfg.quant_linear}` | default_norm: `{cfg.default_norm}`",
            f"- DiT: `{cfg.dit_path}`",
            f"- VAE: `{cfg.vae_path}`",
            f"- Text encoder: `{cfg.text_encoder_path}`",
        ]
    )

    if missing:
        lines.append("")
        lines.append("**Missing files:**")
        lines.extend([f"- `{path}`" for path in missing])

    return "\n".join(lines)
