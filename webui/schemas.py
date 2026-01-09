import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from webui.discovery import discover_checkpoints, infer_from_checkpoint

from webui.utils import check_paths
from webui import preset_loader


DEFAULT_MODEL_ROOT = "/workspace/TurboDiffusion/checkpoints"

# 公共默认路径 - 参数复用
DEFAULT_VAE_PATH = "checkpoints/Wan2.1_VAE.pth"
DEFAULT_TEXT_ENCODER_PATH = "checkpoints/models_t5_umt5-xxl-enc-bf16.pth"


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
        dit_path_high=_resolve_checkpoint_path(cfg.dit_path_high, roots) if cfg.dit_path_high else None,
        model=cfg.model,
        resolution=cfg.resolution,
        aspect_ratio=cfg.aspect_ratio,
        quant_linear=cfg.quant_linear,
        default_norm=cfg.default_norm,
        info=cfg.info,
    )


def create_preset(
    name: str,
    dit_path: str,
    dit_path_high: str = None,
    vae_path: str = None,
    text_encoder_path: str = None,
    model: str = None,
    resolution: str = None,
    aspect_ratio: str = None,
    quant_linear: bool = None,
    default_norm: bool = None,
    info: str = "",
) -> "EngineConfig":
    """智能创建预设配置，自动推断参数并复用公共路径。

    参数：
        name: 预设名称
        dit_path: DiT checkpoint 路径（必需）
        vae_path: VAE 路径（可选，默认使用 DEFAULT_VAE_PATH）
        text_encoder_path: Text encoder 路径（可选，默认使用 DEFAULT_TEXT_ENCODER_PATH）
        model: 模型类型（可选，从 dit_path 自动推断）
        resolution: 分辨率（可选，从 dit_path 自动推断）
        aspect_ratio: 宽高比（可选，从 dit_path 自动推断）
        quant_linear: 是否量化（可选，从 dit_path 自动推断）
        default_norm: 是否使用默认归一化（可选，默认 False）
        info: 额外信息

    返回：
        完整的 EngineConfig 对象
    """
    # 从 dit_path 推断参数
    inferred, auto_info = infer_from_checkpoint(Path(dit_path))

    # 使用提供的参数覆盖推断结果，否则使用推断值或默认值
    return EngineConfig(
        name=name,
        dit_path=dit_path,
        vae_path=vae_path or DEFAULT_VAE_PATH,
        text_encoder_path=text_encoder_path or DEFAULT_TEXT_ENCODER_PATH,
        dit_path_high=dit_path_high,
        model=model or inferred.get("model") or "Wan2.1-1.3B",
        resolution=resolution or inferred.get("resolution") or "480p",
        aspect_ratio=aspect_ratio or inferred.get("aspect_ratio") or "16:9",
        quant_linear=quant_linear if quant_linear is not None else inferred.get("quant_linear", False),
        default_norm=default_norm if default_norm is not None else inferred.get("default_norm", False),
        info=info or auto_info,
    )


@dataclass(frozen=True)
class EngineConfig:
    name: str
    dit_path: str
    vae_path: str
    text_encoder_path: str
    dit_path_high: str | None = None
    model: str = "Wan2.1-1.3B"
    resolution: str = "480p"
    aspect_ratio: str = "16:9"
    quant_linear: bool = True
    default_norm: bool = False
    info: str = ""

# 使用智能工厂函数创建预设 - 自动推断参数并复用公共路径
PRESETS = {
    # Wan2.1 T2V 模型
    "Wan2.1 T2V 1.3B 480p (quant)": create_preset(
        name="Wan2.1 T2V 1.3B 480p (quant)",
        dit_path="checkpoints/TurboWan2.1-T2V-1.3B-480P-quant.pth",
        # quant_linear、model、resolution、aspect_ratio 均从文件名自动推断
        # vae_path 和 text_encoder_path 使用默认公共路径
    ),
    "Wan2.1 T2V 1.3B 480p (fp16)": create_preset(
        name="Wan2.1 T2V 1.3B 480p (fp16)",
        dit_path="checkpoints/TurboWan2.1-T2V-1.3B-480P.pth",
        # 自动推断 quant_linear=False（文件名不含"quant"）
    ),
    "Wan2.1 T2V 14B 720p (quant, 5090 recommended)": create_preset(
        name="Wan2.1 T2V 14B 720p (quant, 5090 recommended)",
        dit_path="checkpoints/TurboWan2.1-T2V-14B-720P-quant.pth",
        # 自动推断：model=Wan2.1-14B, resolution=720p, quant_linear=True
    ),
    "Wan2.1 T2V 14B 720p (fp16, >40GB GPU)": create_preset(
        name="Wan2.1 T2V 14B 720p (fp16, >40GB GPU)",
        dit_path="checkpoints/TurboWan2.1-T2V-14B-720P.pth",
        # 自动推断：model=Wan2.1-14B, resolution=720p, quant_linear=False
    ),
    "Wan2.1 T2V 14B 480p (quant, 5090 recommended)": create_preset(
        name="Wan2.1 T2V 14B 480p (quant, 5090 recommended)",
        dit_path="checkpoints/TurboWan2.1-T2V-14B-480P-quant.pth",
        # 自动推断所有参数
    ),
    "Wan2.1 T2V 14B 480p (fp16, >40GB GPU)": create_preset(
        name="Wan2.1 T2V 14B 480p (fp16, >40GB GPU)",
        dit_path="checkpoints/TurboWan2.1-T2V-14B-480P.pth",
        # 自动推断所有参数
    ),

    # Wan2.2 I2V 模型（A14B 720p）
    #
    # Note: Wan2.2 I2V inference requires *both* low-noise and high-noise DiT
    # checkpoints. We store the low-noise path in ``dit_path`` and the
    # high-noise path in ``dit_path_high``.
    #
    # Upstream naming:
    # - TurboWan2.2-I2V-A14B-low-720P[-quant].pth
    # - TurboWan2.2-I2V-A14B-high-720P[-quant].pth
    "Wan2.2 I2V A14B 720p (quant)": create_preset(
        name="Wan2.2 I2V A14B 720p (quant)",
        dit_path="checkpoints/TurboWan2.2-I2V-A14B-low-720P-quant.pth",
        dit_path_high="checkpoints/TurboWan2.2-I2V-A14B-high-720P-quant.pth",
        # 自动推断：model=Wan2.2-A14B, resolution=720p, quant_linear=True
    ),
    "Wan2.2 I2V A14B 720p (fp16)": create_preset(
        name="Wan2.2 I2V A14B 720p (fp16)",
        dit_path="checkpoints/TurboWan2.2-I2V-A14B-low-720P.pth",
        dit_path_high="checkpoints/TurboWan2.2-I2V-A14B-high-720P.pth",
        # 自动推断：model=Wan2.2-A14B, resolution=720p, quant_linear=False
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

    all_checkpoints: List[Path] = []
    seen_paths: set[str] = set()
    for checkpoints_dir in _discovery_dirs():
        for path in discover_checkpoints(checkpoints_dir).values():
            try:
                resolved = path.resolve()
            except Exception:
                resolved = path
            key = str(resolved)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            all_checkpoints.append(path)

    # Pair Wan2.2 I2V checkpoints: (low/high) must be loaded together.
    i2v_pattern = re.compile(
        r"^TurboWan2\.2-I2V-(?P<size>[A-Za-z0-9]+)-(?P<noise>low|high)-(?P<res>[0-9]+P)(?P<quant>-quant)?$",
        re.IGNORECASE,
    )
    i2v_groups: Dict[str, Dict[str, Path]] = {}
    t2v_paths: List[Path] = []

    for path in all_checkpoints:
        m = i2v_pattern.match(path.stem)
        if not m:
            t2v_paths.append(path)
            continue
        base_stem = f"TurboWan2.2-I2V-{m.group('size')}-{m.group('res')}{m.group('quant') or ''}"
        noise = m.group("noise").lower()
        group = i2v_groups.setdefault(base_stem, {})
        group.setdefault(noise, path)

    for path in t2v_paths:
        fields, info = infer_from_checkpoint(path)

        name_base = str(fields.get("name") or f"Auto: {path.stem}")
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

    for base_stem, parts in sorted(i2v_groups.items()):
        low_path = parts.get("low")
        high_path = parts.get("high")
        if not low_path or not high_path:
            # Avoid exposing half-configured I2V entries that can't run.
            continue

        fields, info = infer_from_checkpoint(low_path)

        name_base = str(fields.get("name") or f"Auto: {base_stem}")
        name = name_base
        idx = 1
        while name in discovered:
            idx += 1
            name = f"{name_base} ({idx})"

        cfg = EngineConfig(
            name=name,
            dit_path=str(low_path),
            dit_path_high=str(high_path),
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
        if cfg.dit_path_high:
            manual_dit_filenames.add(Path(cfg.dit_path_high).name.casefold())

    filtered: List[str] = []
    for preset_name in names:
        if _is_auto_preset_name(preset_name):
            cfg = get_preset(preset_name)
            dit_filename = Path(cfg.dit_path).name.casefold()
            if cfg.dit_path_high:
                high_filename = Path(cfg.dit_path_high).name.casefold()
                if dit_filename in manual_dit_filenames and high_filename in manual_dit_filenames:
                    continue
            else:
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
            *([f"- DiT (high-noise): `{cfg.dit_path_high}`"] if cfg.dit_path_high else []),
            f"- VAE: `{cfg.vae_path}`",
            f"- Text encoder: `{cfg.text_encoder_path}`",
        ]
    )

    if missing:
        lines.append("")
        lines.append("**Missing files:**")
        lines.extend([f"- `{path}`" for path in missing])

    return "\n".join(lines)
