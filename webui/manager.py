import gc
from typing import Optional
from .schemas import EngineConfig
from .engine_wan21 import TurboWanT2VEngine

class EngineManager:
    def __init__(self):
        self.engine: Optional[TurboWanT2VEngine] = None
        self.cfg: Optional[EngineConfig] = None

    def is_loaded(self):
        return self.engine is not None

    def load(self, cfg: EngineConfig):
        if self.engine is not None and self.cfg and self.cfg.name == cfg.name:
            return
        self.unload()
        self.cfg = cfg
        self.engine = TurboWanT2VEngine(cfg)

    def unload(self):
        if self.engine is not None:
            self.engine.close()
        self.engine = None
        self.cfg = None
        gc.collect()