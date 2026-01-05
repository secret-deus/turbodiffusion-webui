import importlib
import sys
import types

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

    def load(self, cfg):
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

        def load(self, cfg):
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

    assert video_path == ""
    assert meta == {}
    assert status.startswith("❌ Error during inference")
    assert "Inference failed" in logs
