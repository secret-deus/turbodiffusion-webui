import torch
from imaginaire.utils import log

from webui.engine_wan21 import TurboWanT2VEngine
from webui.utils import check_paths

class EngineManager:
    def __init__(self):
        self.engine = None
        self.cfg = None
        self.last_error = ""

    def is_loaded(self):
        return self.engine is not None

    def load(self, cfg):
        missing = check_paths(cfg)
        if missing:
            self.last_error = "Missing checkpoints:\n" + "\n".join(missing)
            raise FileNotFoundError(self.last_error)

        # same cfg -> reuse
        if self.engine is not None and self.cfg == cfg:
            return self.engine

        # different cfg -> unload old
        if self.engine is not None:
            self.unload()

        log.info(f"[Manager] Loading engine preset: {cfg.name}")
        self.engine = TurboWanT2VEngine(
            dit_path=cfg.dit_path,
            vae_path=cfg.vae_path,
            text_encoder_path=cfg.text_encoder_path,
            model=cfg.model,
            resolution=cfg.resolution,
            aspect_ratio=cfg.aspect_ratio,
            quant_linear=cfg.quant_linear,
            default_norm=cfg.default_norm,
            keep_dit_on_gpu=True,
        )
        self.cfg = cfg
        self.last_error = ""
        return self.engine

    def unload(self):
        log.info("[Manager] Unloading engine ...")
        self.engine = None
        self.cfg = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        log.success("[Manager] Unloaded.")
