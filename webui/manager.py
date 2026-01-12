import torch
from imaginaire.utils import log

from pathlib import Path
import re

from webui.engine_wan21 import TurboWanT2VEngine
from webui.engine_wan22_i2v import TurboWanI2VEngine
from webui.utils import check_paths

class EngineManager:
    def __init__(self):
        self.engine = None
        self.cfg = None
        self.load_opts = None
        self.last_error = ""

    def is_loaded(self):
        return self.engine is not None

    def load(self, cfg, attention_type=None, sla_topk=None, default_norm=None):
        missing = check_paths(cfg)
        if missing:
            self.last_error = "Missing checkpoints:\n" + "\n".join(missing)
            raise FileNotFoundError(self.last_error)

        model_name = str(getattr(cfg, "model", "") or "")
        is_wan22 = model_name.startswith("Wan2.2")

        effective_attention_type = attention_type or "sla"
        effective_sla_topk = round(float(sla_topk), 4) if sla_topk is not None else 0.1
        effective_default_norm = bool(default_norm) if default_norm is not None else bool(cfg.default_norm)

        load_opts = {
            "attention_type": str(effective_attention_type),
            "sla_topk": float(effective_sla_topk),
            "default_norm": bool(effective_default_norm),
        }

        # same cfg -> reuse
        if self.engine is not None and self.cfg == cfg and self.load_opts == load_opts:
            return self.engine

        # different cfg -> unload old
        if self.engine is not None:
            self.unload()

        log.info(f"[Manager] Loading engine preset: {cfg.name}")
        log.info(
            f"[Manager] Load opts: attention_type={load_opts['attention_type']} "
            f"sla_topk={load_opts['sla_topk']} default_norm={load_opts['default_norm']}"
        )

        if is_wan22:
            if not getattr(cfg, "dit_path_high", None):
                raise FileNotFoundError(
                    "Wan2.2 I2V requires both low-noise and high-noise checkpoints. "
                    "Missing `dit_path_high`."
                )

            low_path = cfg.dit_path
            high_path = cfg.dit_path_high
            if re.search(r"(?i)-high-", Path(low_path).stem) and re.search(r"(?i)-low-", Path(high_path).stem):
                low_path, high_path = high_path, low_path

            self.engine = TurboWanI2VEngine(
                low_noise_model_path=low_path,
                high_noise_model_path=high_path,
                vae_path=cfg.vae_path,
                text_encoder_path=cfg.text_encoder_path,
                model=cfg.model,
                resolution=cfg.resolution,
                aspect_ratio=cfg.aspect_ratio,
                attention_type=load_opts["attention_type"],
                sla_topk=load_opts["sla_topk"],
                quant_linear=cfg.quant_linear,
                default_norm=load_opts["default_norm"],
            )
        else:
            self.engine = TurboWanT2VEngine(
                dit_path=cfg.dit_path,
                vae_path=cfg.vae_path,
                text_encoder_path=cfg.text_encoder_path,
                model=cfg.model,
                resolution=cfg.resolution,
                aspect_ratio=cfg.aspect_ratio,
                quant_linear=cfg.quant_linear,
                default_norm=load_opts["default_norm"],
                attention_type=load_opts["attention_type"],
                sla_topk=load_opts["sla_topk"],
                keep_dit_on_gpu=True,
            )
        self.cfg = cfg
        self.load_opts = load_opts
        self.last_error = ""
        return self.engine

    def unload(self):
        log.info("[Manager] Unloading engine ...")
        self.engine = None
        self.cfg = None
        self.load_opts = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        log.success("[Manager] Unloaded.")
