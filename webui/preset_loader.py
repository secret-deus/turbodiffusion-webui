import os
from pathlib import Path
from typing import Dict, Iterable, Type

DEFAULT_CKPT_DIR = "/workspace/TurboDiffusion/checkpoints"


def _get_search_dirs() -> Iterable[Path]:
    env_paths = os.environ.get("MODEL_PATHS") or os.environ.get("CKPT_DIR") or DEFAULT_CKPT_DIR
    parts = [p for p in env_paths.split(os.pathsep) if p]
    seen = set()
    for part in parts:
        path = Path(part).expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        yield path


def _find_required_file(filename: str, preferred_dir: Path, search_dirs: Iterable[Path]) -> Path | None:
    candidate = preferred_dir / filename
    if candidate.exists():
        return candidate
    for directory in search_dirs:
        candidate = directory / filename
        if candidate.exists():
            return candidate
    return None


def _derive_model(stem: str) -> str:
    if "14b" in stem.lower():
        return "Wan2.1-14B"
    if "1.3b" in stem.lower():
        return "Wan2.1-1.3B"
    return "Wan2.1-1.3B"


def _derive_resolution(stem: str) -> str:
    if "720" in stem:
        return "720p"
    if "1080" in stem:
        return "1080p"
    return "480p"


def discover_presets(base_presets: Dict[str, object], engine_config_type: Type[object]) -> Dict[str, object]:
    """Discover checkpoints and build EngineConfig entries.

    Args:
        base_presets: Existing presets to treat as defaults/fallbacks.
        engine_config_type: The EngineConfig class used to construct configs.

    Returns:
        A merged mapping of preset names to EngineConfig instances.
    """
    merged: Dict[str, object] = dict(base_presets)
    search_dirs = list(_get_search_dirs())

    for directory in search_dirs:
        if not directory.exists():
            continue
        for ckpt_path in directory.glob("TurboWan*.pth"):
            stem = ckpt_path.stem
            vae_path = _find_required_file("Wan2.1_VAE.pth", ckpt_path.parent, search_dirs)
            text_enc_path = _find_required_file("models_t5_umt5-xxl-enc-bf16.pth", ckpt_path.parent, search_dirs)
            if not vae_path or not text_enc_path:
                continue

            name_base = f"Auto | {stem}"
            name = name_base
            idx = 1
            while name in merged:
                idx += 1
                name = f"{name_base} ({idx})"

            merged[name] = engine_config_type(
                name=name,
                dit_path=str(ckpt_path),
                vae_path=str(vae_path),
                text_encoder_path=str(text_enc_path),
                model=_derive_model(stem),
                resolution=_derive_resolution(stem),
                aspect_ratio="16:9",
                quant_linear="quant" in stem.lower(),
                default_norm=False,
            )

    return merged
