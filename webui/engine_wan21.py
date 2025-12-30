import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

import torch
from einops import rearrange, repeat
from tqdm import tqdm

from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface

# TurboDiffusion repo 内置
from turbodiffusion.inference.modify_model import tensor_kwargs, create_model

# ---------- imaginaire fallback ----------
def _save_video_fallback(tensor_c_t_h_w: torch.Tensor, save_path: str, fps: int = 16):
    import imageio
    c, t, h, w = tensor_c_t_h_w.shape
    video = (tensor_c_t_h_w.permute(1, 2, 3, 0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    imageio.mimwrite(save_path, list(video), fps=fps)

try:
    from imaginaire.utils.io import save_image_or_video as _save_image_or_video
except Exception:
    _save_image_or_video = None

def save_image_or_video(t, path, fps=16):
    if _save_image_or_video is not None:
        return _save_image_or_video(t, path, fps=fps)
    return _save_video_fallback(t, path, fps=fps)

# ---------- spargeattn detection ----------
def _has_spargeattn():
    try:
        import spargeattn  # noqa: F401
        return True
    except Exception:
        return False

@dataclass(frozen=True)
class Wan21Config:
    name: str
    dit_path: str
    vae_path: str
    text_encoder_path: str
    model: str = "Wan2.1-1.3B"
    resolution: str = "480p"
    aspect_ratio: str = "16:9"
    quant_linear: bool = True
    default_norm: bool = False

class TurboWanT2VEngine:
    def __init__(self, cfg: Wan21Config):
        self.cfg = cfg
        self.device = tensor_kwargs["device"]
        self.dtype = tensor_kwargs["dtype"]

        # attention fallback
        self.has_sparge = _has_spargeattn()
        self.net = None
        self.tokenizer = None

        # store arguments for create_model
        self.args = SimpleNamespace(
            dit_path=cfg.dit_path,
            model=cfg.model,
            num_steps=4,
            sigma_max=80,
            vae_path=cfg.vae_path,
            text_encoder_path=cfg.text_encoder_path,
            num_frames=81,
            prompt="",
            resolution=cfg.resolution,
            aspect_ratio=cfg.aspect_ratio,
            seed=0,
            save_path="",
            attention_type="sla",
            sla_topk=0.10,
            quant_linear=cfg.quant_linear,
            default_norm=cfg.default_norm,
        )

        self._load_models()

    def _log(self, cb, msg):
        if cb:
            cb(msg)

    def _load_models(self):
        # Load DiT
        self.net = create_model(dit_path=self.cfg.dit_path, args=self.args).cpu().eval()
        torch.cuda.empty_cache()

        # VAE tokenizer
        self.tokenizer = Wan2pt1VAEInterface(vae_pth=self.cfg.vae_path)

    def close(self):
        if self.net is not None:
            self.net.cpu()
        self.net = None
        self.tokenizer = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        num_steps: int = 4,
        num_frames: int = 81,
        num_samples: int = 1,
        seed: int = 0,
        attention_type: str = "sla",
        sla_topk: float = 0.10,
        sigma_max: float = 80,
        default_norm: bool = False,
        save_path: str = "outputs/out.mp4",
        fps: int = 16,
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
        log_cb: Optional[Callable[[str], None]] = None,
        use_tqdm: bool = False,
    ) -> str:
        # sagesla requires spargeattn
        if attention_type == "sagesla" and not self.has_sparge:
            self._log(log_cb, "[Engine] SpargeAttn not installed, fallback sagesla -> sla")
            attention_type = "sla"

        # update args
        self.args.prompt = prompt
        self.args.num_steps = int(num_steps)
        self.args.num_frames = int(num_frames)
        self.args.attention_type = attention_type
        self.args.sla_topk = float(sla_topk)
        self.args.sigma_max = float(sigma_max)
        self.args.default_norm = bool(default_norm)

        # get text embedding
        self._log(log_cb, f"[Engine] Encode prompt: {prompt[:80]}...")
        text_emb = get_umt5_embedding(checkpoint_path=self.cfg.text_encoder_path, prompts=prompt).to(**tensor_kwargs)
        clear_umt5_memory()

        condition = {"crossattn_emb": repeat(text_emb.to(**tensor_kwargs), "b l d -> (k b) l d", k=num_samples)}
        w, h = VIDEO_RES_SIZE_INFO[self.cfg.resolution][self.cfg.aspect_ratio]

        state_shape = [
            self.tokenizer.latent_ch,
            self.tokenizer.get_latent_num_frames(num_frames),
            h // self.tokenizer.spatial_compression_factor,
            w // self.tokenizer.spatial_compression_factor,
        ]

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        init_noise = torch.randn(
            num_samples,
            *state_shape,
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        )

        # mid_t schedule
        mid_t = [1.5, 1.4, 1.0][: num_steps - 1]
        t_steps = torch.tensor([math.atan(sigma_max), *mid_t, 0], dtype=torch.float64, device=init_noise.device)
        t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

        x = init_noise.to(torch.float64) * t_steps[0]
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        total_steps = t_steps.shape[0] - 1

        self.net.cuda()

        step_iter = zip(t_steps[:-1], t_steps[1:])
        if use_tqdm:
            step_iter = tqdm(list(step_iter), desc="Sampling", total=total_steps)

        for i, (t_cur, t_next) in enumerate(step_iter, start=1):
            if progress_cb:
                progress_cb("sampling", i, total_steps)

            v_pred = self.net(
                x_B_C_T_H_W=x.to(**tensor_kwargs),
                timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
                **condition,
            ).to(torch.float64)

            x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                *x.shape,
                dtype=torch.float32,
                device=self.device,
                generator=generator,
            )

        samples = x.float()
        self.net.cpu()
        torch.cuda.empty_cache()

        if progress_cb:
            progress_cb("decode", 1, 1)

        video = self.tokenizer.decode(samples)

        to_show = (1.0 + video.clamp(-1, 1)) / 2.0
        out_tensor = rearrange(to_show, "b c t h w -> c t h w")

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        save_image_or_video(out_tensor, save_path, fps=fps)
        self._log(log_cb, f"[Engine] Saved to: {save_path}")

        return save_path