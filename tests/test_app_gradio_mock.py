import importlib
import sys
import types
from pathlib import Path

import pytest


class DummyEngine:
    def generate(self, **_: object) -> str:  # pragma: no cover - behavior checked via raise
        raise RuntimeError("mock failure")


class DummyManager:
    def __init__(self):
        self.engine = None
        self.cfg = None

    def is_loaded(self) -> bool:
        return self.engine is not None

    def load(self, cfg, **_kwargs):
        # simulate successful load without touching real checkpoints
        self.engine = DummyEngine()
        self.cfg = cfg
        return self.engine


DEFAULT_ARGS = {
    "prompt": "test prompt",
    "num_steps": 4,
    "num_frames": 17,
    "num_samples": 1,
    "seed_mode": "fixed",
    "seed": 123,
    "attention_type": "sla",
    "sla_topk": 0.1,
    "sigma_max": 80,
    "fps": 16,
    "keep_dit_on_gpu": True,
    "keep_text_encoder": False,
    "default_norm": False,
}


@pytest.fixture(autouse=True)
def gradio_stub(monkeypatch):
    stub = types.SimpleNamespace(update=lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, "gradio", stub)
    torch_stub = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            current_device=lambda: 0,
            get_device_name=lambda _idx=None: "mock-gpu",
            mem_get_info=lambda: (0, 0),
            max_memory_allocated=lambda: 0,
            empty_cache=lambda: None,
            reset_peak_memory_stats=lambda: None,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "imaginaire", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "imaginaire.utils",
        types.SimpleNamespace(log=types.SimpleNamespace(info=lambda *a, **k: None, success=lambda *a, **k: None)),
    )
    class _StubEngineManager:
        def __init__(self):
            self.engine = None
            self.cfg = None

        def is_loaded(self):
            return False

        def load(self, cfg, **_kwargs):
            self.cfg = cfg
            return None

        def unload(self):
            return None

    monkeypatch.setitem(sys.modules, "webui.engine_wan21", types.SimpleNamespace(TurboWanT2VEngine=object))
    monkeypatch.setitem(sys.modules, "webui.manager", types.SimpleNamespace(EngineManager=_StubEngineManager))
    yield
    sys.modules.pop("gradio", None)
    sys.modules.pop("torch", None)
    sys.modules.pop("imaginaire", None)
    sys.modules.pop("imaginaire.utils", None)
    sys.modules.pop("webui.engine_wan21", None)
    sys.modules.pop("webui.manager", None)


def test_generate_video_surfaces_inference_error(monkeypatch, tmp_path):
    # reload module to pick up stubbed gradio
    sys.modules.pop("app_gradio", None)
    app_gradio = importlib.import_module("app_gradio")

    monkeypatch.setattr(app_gradio, "OUT", tmp_path)
    monkeypatch.setattr(app_gradio, "MANAGER", DummyManager())

    preset_name = app_gradio.PRESET_CHOICES[0]
    video_path, status, logs, meta = app_gradio.generate_video(preset_name, **DEFAULT_ARGS)

    assert video_path is None
    assert meta == {}
    assert status.startswith("❌ Error during inference")
    assert "Inference failed" in logs


def test_generate_video_requires_init_image_for_i2v(monkeypatch, tmp_path):
    sys.modules.pop("app_gradio", None)
    app_gradio = importlib.import_module("app_gradio")

    monkeypatch.setattr(app_gradio, "OUT", tmp_path)

    i2v_presets = [name for name in app_gradio.PRESET_CHOICES if "I2V" in name.upper()]
    assert i2v_presets, "Expected at least one I2V preset in PRESET_CHOICES"
    preset_name = i2v_presets[0]

    video_path, status, logs, meta = app_gradio.generate_video(preset_name, init_image=None, **DEFAULT_ARGS)

    assert video_path is None
    assert meta == {}
    assert "no init image" in status.lower()
    assert "no init image" in logs.lower()


def test_generate_video_passes_init_image_for_i2v(monkeypatch, tmp_path):
    import numpy as np

    class DummyEngineOk:
        def __init__(self):
            self.keep_dit_on_gpu = True
            self.keep_text_encoder = False
            self.last_kwargs = None

        def generate(self, **kwargs):
            self.last_kwargs = dict(kwargs)
            save_path = Path(kwargs["save_path"])
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.touch()
            return str(save_path)

    class DummyManagerOk:
        def __init__(self, engine):
            self.engine = None
            self.cfg = None
            self._engine = engine

        def is_loaded(self) -> bool:
            return False

        def load(self, cfg, **_kwargs):
            self.cfg = cfg
            self.engine = self._engine
            return self.engine

    sys.modules.pop("app_gradio", None)
    app_gradio = importlib.import_module("app_gradio")

    engine = DummyEngineOk()
    monkeypatch.setattr(app_gradio, "MANAGER", DummyManagerOk(engine))
    monkeypatch.setattr(app_gradio, "OUT", tmp_path)

    i2v_presets = [name for name in app_gradio.PRESET_CHOICES if "I2V" in name.upper()]
    assert i2v_presets, "Expected at least one I2V preset in PRESET_CHOICES"
    preset_name = i2v_presets[0]

    init_image = np.zeros((64, 64, 3), dtype=np.uint8)
    video_path, status, logs, meta = app_gradio.generate_video(
        preset_name,
        init_image=init_image,
        i2v_adaptive_resolution=True,
        i2v_boundary=0.9,
        i2v_ode=True,
        **DEFAULT_ARGS,
    )

    assert engine.last_kwargs is not None
    assert engine.last_kwargs["init_image"] is init_image
    assert engine.last_kwargs["adaptive_resolution"] is True
    assert engine.last_kwargs["boundary"] == 0.9
    assert engine.last_kwargs["ode"] is True

    assert video_path
    assert Path(video_path).exists()
    assert "i2v_" in Path(video_path).name
    assert status.startswith("✅ Done")
    assert meta.get("mode") == "i2v"
    assert meta.get("adaptive_resolution") is True
    assert meta.get("boundary") == 0.9
    assert meta.get("ode") is True
