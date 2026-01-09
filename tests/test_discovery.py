import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault(
    "torch",
    types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            current_device=lambda: 0,
            get_device_name=lambda _idx=None: "mock-gpu",
            mem_get_info=lambda: (0, 0),
            max_memory_allocated=lambda: 0,
            empty_cache=lambda: None,
            reset_peak_memory_stats=lambda: None,
        )
    ),
)

from webui.discovery import discover_checkpoints, infer_from_checkpoint
from webui.schemas import _build_discovered_presets
import webui.schemas as schemas


def test_infer_from_checkpoint_parses_filename_tokens(tmp_path):
    ckpt = tmp_path / "TurboWan2.1-T2V-1.3B-480P-quant.pth"
    ckpt.touch()

    fields, info = infer_from_checkpoint(ckpt)

    assert fields["quant_linear"] is True
    assert fields["resolution"] == "480p"
    assert fields["aspect_ratio"] == "16:9"
    assert fields["model"] == "Wan2.1-1.3B"
    assert "model=Wan2.1-1.3B" in info


def test_infer_from_checkpoint_parses_wan22_i2v_tokens(tmp_path):
    ckpt = tmp_path / "TurboWan2.2-I2V-A14B-high-720P-quant.pth"
    ckpt.touch()

    fields, info = infer_from_checkpoint(ckpt)

    assert fields["quant_linear"] is True
    assert fields["resolution"] == "720p"
    assert fields["aspect_ratio"] == "16:9"
    assert fields["model"] == "Wan2.2-A14B"
    assert "model=Wan2.2-A14B" in info


def test_get_preset_infers_wan22_i2v_pair(tmp_path, monkeypatch):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()

    # shared deps
    (checkpoints_dir / "Wan2.1_VAE.pth").touch()
    (checkpoints_dir / "models_t5_umt5-xxl-enc-bf16.pth").touch()

    low = checkpoints_dir / "TurboWan2.2-I2V-A14B-low-720P-quant.pth"
    high = checkpoints_dir / "TurboWan2.2-I2V-A14B-high-720P-quant.pth"
    low.touch()
    high.touch()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODEL_PATHS", str(checkpoints_dir))

    # simulate legacy preset that only points to one checkpoint
    monkeypatch.setattr(
        schemas,
        "PRESETS",
        {
            "Legacy I2V": schemas.EngineConfig(
                name="Legacy I2V",
                dit_path="checkpoints/TurboWan2.2-I2V-A14B-low-720P-quant.pth",
                vae_path="checkpoints/Wan2.1_VAE.pth",
                text_encoder_path="checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
                model="Wan2.2-A14B",
                resolution="720p",
                aspect_ratio="16:9",
                quant_linear=True,
                default_norm=False,
            )
        },
    )

    cfg = schemas.get_preset("Legacy I2V")

    assert cfg.dit_path_high is not None
    assert Path(cfg.dit_path).name == low.name
    assert Path(cfg.dit_path_high).name == high.name


def test_infer_from_checkpoint_applies_sidecar_overrides(tmp_path):
    ckpt = tmp_path / "TurboWan2.1-T2V-14B-720P-quant.pth"
    ckpt.touch()
    sidecar = ckpt.with_suffix(".json")
    sidecar.write_text(
        """
        {
            "name": "Custom 14B",
            "model": "Wan2.1-14B",
            "resolution": "720p",
            "aspect_ratio": "16:9",
            "quant_linear": false,
            "default_norm": true,
            "vae_path": "custom_vae.pth"
        }
        """,
        encoding="utf-8",
    )

    fields, info = infer_from_checkpoint(ckpt)

    assert fields["name"] == "Custom 14B"
    assert fields["quant_linear"] is False  # override
    assert fields["default_norm"] is True
    assert fields["vae_path"] == "custom_vae.pth"
    assert "overrides" in info


def test_build_discovered_presets_uses_inferred_defaults(tmp_path, monkeypatch):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    ckpt = checkpoints_dir / "TurboWan2.1-T2V-1.3B-480P-quant.pth"
    ckpt.touch()

    # ensure relative path lookup matches the temporary directory
    monkeypatch.chdir(tmp_path)

    presets = _build_discovered_presets()

    assert "Auto: TurboWan2.1-T2V-1.3B-480P-quant" in presets
    cfg = presets["Auto: TurboWan2.1-T2V-1.3B-480P-quant"]
    assert cfg.quant_linear is True
    assert cfg.resolution == "480p"
    assert cfg.model == "Wan2.1-1.3B"
    assert cfg.dit_path.endswith(ckpt.name)
    assert Path(cfg.vae_path).name == "Wan2.1_VAE.pth"


def test_discover_checkpoints_filters_to_wan21(tmp_path):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    good = checkpoints_dir / "TurboWan2.1-T2V-1.3B-480P-quant.pth"
    bad = checkpoints_dir / "other-model.pth"
    good.touch()
    bad.touch()

    discovered = discover_checkpoints(checkpoints_dir)
    assert good.stem in discovered
    assert bad.stem not in discovered


def test_discoverable_preset_names_hides_duplicate_auto_presets(tmp_path, monkeypatch):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_host_dir = tmp_path / "checkpoints_host"
    checkpoints_dir.mkdir()
    checkpoints_host_dir.mkdir()

    # shared deps exist in image checkpoints dir
    (checkpoints_dir / "Wan2.1_VAE.pth").touch()
    (checkpoints_dir / "models_t5_umt5-xxl-enc-bf16.pth").touch()

    # DiT weights: one in host mount, one in image directory
    (checkpoints_host_dir / "TurboWan2.1-T2V-1.3B-480P-quant.pth").touch()
    (checkpoints_dir / "TurboWan2.1-T2V-14B-720P-quant.pth").touch()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODEL_PATHS", f"{checkpoints_host_dir},{checkpoints_dir}")

    manual_presets = {
        "Wan2.1 T2V 1.3B 480p (quant)": schemas.EngineConfig(
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
        "Wan2.1 T2V 14B 720p (quant)": schemas.EngineConfig(
            name="Wan2.1 T2V 14B 720p (quant)",
            dit_path="checkpoints/TurboWan2.1-T2V-14B-720P-quant.pth",
            vae_path="checkpoints/Wan2.1_VAE.pth",
            text_encoder_path="checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
            model="Wan2.1-14B",
            resolution="720p",
            aspect_ratio="16:9",
            quant_linear=True,
            default_norm=False,
        ),
    }

    discovered_presets = schemas._build_discovered_presets()
    monkeypatch.setattr(schemas, "PRESETS", {**manual_presets, **discovered_presets})

    names = schemas.discoverable_preset_names()

    assert names == list(manual_presets.keys())
    assert all(not preset.startswith("Auto:") for preset in names)
