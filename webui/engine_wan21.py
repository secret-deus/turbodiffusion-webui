import math
import time
from pathlib import Path
from typing import Optional, Union, Callable

import torch
from einops import rearrange, repeat
from tqdm import tqdm

from imaginaire.utils.io import save_image_or_video
from imaginaire.utils import log

from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface

# ✅ 关键：绝对导入，避免 webui/ 下找不到 modify_model
from turbodiffusion.inference.modify_model import tensor_kwargs, create_model


ProgressCB = Optional[Callable[[str, int, int], None]]
LogCB = Optional[Callable[[str], None]]


class TurboWanT2VEngine:
    """
    Load-once engine for Wan2.1 T2V TurboDiffusion inference.
    Rebuild from wan2.1_t2v_infer.py main loop:
      - load VAE once
      - load DiT once
      - compute UMT5 embedding per prompt, then clear UMT5 memory (default)
      - do rCM sampling (1~4 steps)
      - decode VAE
      - save mp4
    """

    def __init__(
        self,
        dit_path: str,
        vae_path: str = "checkpoints/Wan2.1_VAE.pth",
        text_encoder_path: str = "checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
        model: str = "Wan2.1-1.3B",
        resolution: str = "480p",
        aspect_ratio: str = "16:9",
        sigma_max: float = 80.0,
        attention_type: str = "sla",
        sla_topk: float = 0.1,
        quant_linear: bool = False,
        default_norm: bool = False,
        keep_dit_on_gpu: bool = True,
        keep_text_encoder: bool = False,
    ):
        # config
        self.dit_path = dit_path
        self.vae_path = vae_path
        self.text_encoder_path = text_encoder_path

        self.model = model
        self.resolution = resolution
        self.aspect_ratio = aspect_ratio
        self.sigma_max = sigma_max
        self.attention_type = attention_type
        self.sla_topk = sla_topk
        self.quant_linear = quant_linear
        self.default_norm = default_norm

        self.keep_dit_on_gpu = keep_dit_on_gpu
        self.keep_text_encoder = keep_text_encoder

        # 1) Load VAE tokenizer/interface once
        log.info(f"[Engine] Loading VAE: {vae_path}")
        self.tokenizer = Wan2pt1VAEInterface(vae_pth=vae_path)
        log.success("[Engine] VAE loaded.")

        # 2) Build args-like object for create_model
        self.args = self._make_args_namespace()

        # 3) Load DiT once (CPU first)
        log.info(f"[Engine] Loading DiT: {dit_path}")
        self.net = create_model(dit_path=dit_path, args=self.args).cpu().eval()
        torch.cuda.empty_cache()
        log.success("[Engine] DiT loaded.")

        self._warmed_up = False

    # -----------------------------
    # args shim for create_model
    # -----------------------------
    def _make_args_namespace(self):
        """
        create_model expects an argparse-like args with certain fields.
        We'll create a lightweight object with these attributes.
        """
        class Args:
            pass

        args = Args()
        args.dit_path = self.dit_path
        args.model = self.model
        args.num_samples = 1
        args.num_steps = 4
        args.sigma_max = self.sigma_max
        args.vae_path = self.vae_path
        args.text_encoder_path = self.text_encoder_path
        args.num_frames = 81
        args.prompt = ""
        args.resolution = self.resolution
        args.aspect_ratio = self.aspect_ratio
        args.seed = 0
        args.save_path = ""
        args.attention_type = self.attention_type
        args.sla_topk = self.sla_topk
        args.quant_linear = self.quant_linear
        args.default_norm = self.default_norm
        return args

    def _update_args(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self.args, k, v)

    # -----------------------------
    # lifecycle helpers
    # -----------------------------
    def warmup(self):
        """
        Warmup once to reduce first inference latency.
        """
        if self._warmed_up:
            return
        try:
            log.info("[Engine] Warmup start...")
            _ = self.generate(
                prompt="warmup",
                num_steps=1,
                num_frames=17,
                seed=0,
                num_samples=1,
                save_path=None,
                return_tensor=False,
                progress_cb=None,
                log_cb=None,
                use_tqdm=False,
            )
            self._warmed_up = True
            log.success("[Engine] Warmup done.")
        except Exception as e:
            log.warning(f"[Engine] Warmup failed (ignored): {e}")

    def unload_dit(self):
        """
        Optional: explicitly move DiT to CPU and clear cache.
        """
        try:
            self.net.cpu()
        except Exception:
            pass
        torch.cuda.empty_cache()

    # -----------------------------
    # main generate
    # -----------------------------
    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        num_steps: int = 4,
        num_frames: int = 81,
        seed: int = 0,
        num_samples: int = 1,
        init_image: Optional[object] = None,
        i2v_strength: float = 1.0,
        resolution: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        sigma_max: Optional[float] = None,
        attention_type: Optional[str] = None,
        sla_topk: Optional[float] = None,
        quant_linear: Optional[bool] = None,
        default_norm: Optional[bool] = None,
        save_path: Optional[Union[str, Path]] = None,
        fps: int = 16,
        return_tensor: bool = False,
        progress_cb: ProgressCB = None,
        log_cb: LogCB = None,
        use_tqdm: bool = False,
    ):
        """
        Run inference and optionally save mp4.

        Returns:
          - if save_path is provided: str(save_path)
          - else: video numpy/tensor (C,T,H,W) in [0,1]
        """

        # ---------- resolve defaults ----------
        if resolution is None:
            resolution = self.resolution
        if aspect_ratio is None:
            aspect_ratio = self.aspect_ratio
        if sigma_max is None:
            sigma_max = self.sigma_max
        if attention_type is None:
            attention_type = self.attention_type
        if sla_topk is None:
            sla_topk = self.sla_topk
        if quant_linear is None:
            quant_linear = self.quant_linear
        if default_norm is None:
            default_norm = self.default_norm

        # update args mirror (for create_model compat / net behavior)
        self._update_args(
            prompt=prompt,
            num_steps=num_steps,
            num_frames=num_frames,
            seed=seed,
            num_samples=num_samples,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            sigma_max=sigma_max,
            attention_type=attention_type,
            sla_topk=sla_topk,
            quant_linear=quant_linear,
            default_norm=default_norm,
        )

        # ---------- stage: embedding ----------
        if log_cb:
            log_cb(f"[Engine] Embedding prompt: {prompt}")
        if progress_cb:
            progress_cb("embedding", 0, 1)

        # IMPORTANT: your original script uses:
        # text_emb = get_umt5_embedding(...).to(**tensor_kwargs)
        text_emb = get_umt5_embedding(
            checkpoint_path=self.text_encoder_path,
            prompts=prompt
        ).to(**tensor_kwargs)

        # match script: clear UMT5 memory immediately
        if not self.keep_text_encoder:
            clear_umt5_memory()

        if progress_cb:
            progress_cb("embedding", 1, 1)

        # ---------- stage: setup ----------
        if log_cb:
            log_cb("[Engine] Preparing condition + noise...")
        if progress_cb:
            progress_cb("setup", 0, 1)

        condition = {
            "crossattn_emb": repeat(
                text_emb.to(**tensor_kwargs),
                "b l d -> (k b) l d",
                k=num_samples
            )
        }

        w, h = VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]

        state_shape = [
            self.tokenizer.latent_ch,
            self.tokenizer.get_latent_num_frames(num_frames),
            h // self.tokenizer.spatial_compression_factor,
            w // self.tokenizer.spatial_compression_factor,
        ]

        generator = torch.Generator(device=tensor_kwargs["device"])
        generator.manual_seed(seed)

        init_latent = None
        init_latent_noise = None
        if init_image is not None:
            if log_cb:
                log_cb(f"[Engine] I2V enabled: strength={i2v_strength}")
            init_latent = self._encode_init_image(
                init_image=init_image,
                width=w,
                height=h,
                device=tensor_kwargs["device"],
                log_cb=log_cb,
            ).to(dtype=torch.float64)
            if init_latent.ndim != 5:
                raise ValueError(f"init_latent must be 5D (B,C,T,H,W), got shape={tuple(init_latent.shape)}")

            # repeat condition to match num_samples
            if init_latent.size(0) == 1 and num_samples > 1:
                init_latent = init_latent.repeat(num_samples, 1, 1, 1, 1)
            if init_latent.size(0) != num_samples:
                raise ValueError(
                    f"init_latent batch={init_latent.size(0)} does not match num_samples={num_samples}"
                )

            # deterministic per-seed noise for conditioning (inpainting-style)
            cond_gen = torch.Generator(device=tensor_kwargs["device"])
            cond_gen.manual_seed(seed + 1_000_003)
            init_latent_noise = torch.randn(
                init_latent.size(0),
                init_latent.size(1),
                init_latent.size(2),
                init_latent.size(3),
                init_latent.size(4),
                dtype=torch.float64,
                device=tensor_kwargs["device"],
                generator=cond_gen,
            )

        init_noise = torch.randn(
            num_samples,
            *state_shape,
            dtype=torch.float32,
            device=tensor_kwargs["device"],
            generator=generator,
        )

        # timesteps (same as script)
        # mid_t = [1.5, 1.4, 1.0][: num_steps - 1]
        mid_t = [1.5, 1.4, 1.0][: max(num_steps - 1, 0)]

        t_steps = torch.tensor(
            [math.atan(sigma_max), *mid_t, 0],
            dtype=torch.float64,
            device=init_noise.device,
        )
        # convert TrigFlow timesteps to RectifiedFlow
        t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

        if progress_cb:
            progress_cb("setup", 1, 1)

        # ---------- stage: sampling ----------
        total_steps = t_steps.shape[0] - 1
        if log_cb:
            log_cb(f"[Engine] Sampling: steps={num_steps} (total loops={total_steps})")
        if progress_cb:
            progress_cb("sampling", 0, total_steps)

        x = init_noise.to(torch.float64) * t_steps[0]

        if init_latent is not None:
            # Apply first-frame conditioning in latent space (masking / inpainting trick).
            # For rectified flow, a common approximation is x_t = (1 - t) * x0 + t * eps.
            def _apply_first_frame_condition(x_B_C_T_H_W: torch.Tensor, t: torch.Tensor) -> None:
                t_scalar = float(t.item()) if isinstance(t, torch.Tensor) else float(t)
                strength = float(i2v_strength) if i2v_strength is not None else 1.0
                strength = max(0.0, min(1.0, strength))
                t_eff = max(0.0, min(1.0, t_scalar * strength))

                x_cond = (1.0 - t_eff) * init_latent + t_eff * init_latent_noise
                # only constrain the first latent frame
                x_B_C_T_H_W[:, :, 0:1, :, :] = x_cond[:, :, 0:1, :, :]

            _apply_first_frame_condition(x, t_steps[0])
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)

        # move net to GPU
        self.net.cuda()

        it = list(zip(t_steps[:-1], t_steps[1:]))
        if use_tqdm:
            it = tqdm(it, desc="Sampling", total=total_steps)

        for i, (t_cur, t_next) in enumerate(it):
            v_pred = self.net(
                x_B_C_T_H_W=x.to(**tensor_kwargs),
                timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
                **condition
            ).to(torch.float64)

            x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                *x.shape,
                dtype=torch.float32,
                device=tensor_kwargs["device"],
                generator=generator,
            )

            if init_latent is not None:
                _apply_first_frame_condition(x, t_next)

            if progress_cb:
                progress_cb("sampling", i + 1, total_steps)

        samples = x.float()

        # keep net on GPU or free
        if not self.keep_dit_on_gpu:
            self.net.cpu()
            torch.cuda.empty_cache()

        # ---------- stage: decode ----------
        if log_cb:
            log_cb("[Engine] Decoding VAE...")
        if progress_cb:
            progress_cb("decode", 0, 1)

        video = self.tokenizer.decode(samples)  # (B,C,T,H,W) usually

        if progress_cb:
            progress_cb("decode", 1, 1)

        # ---------- stage: normalize + format ----------
        # original script:
        # to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0
        # save_image_or_video(rearrange(to_show, "n b c t h w -> c t (n h) (b w)"), ...)
        #
        # Here: we have single batch 'video' (B,C,T,H,W)
        to_show = (1.0 + video.clamp(-1, 1)) / 2.0  # [0,1]
        to_show = to_show.float().cpu()

        # tile multiple samples horizontally like original script:
        # output shape for save_image_or_video: (C,T,H,W)
        # We'll tile B samples across width: (C,T,H,B*W)
        to_show = rearrange(to_show, "b c t h w -> c t h (b w)")

        # ---------- stage: save ----------
        if save_path is not None:
            if log_cb:
                log_cb(f"[Engine] Saving video: {save_path}")
            if progress_cb:
                progress_cb("save", 0, 1)

            save_path = str(save_path)
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            save_image_or_video(to_show, save_path, fps=fps)

            if progress_cb:
                progress_cb("save", 1, 1)

            return save_path

        # return raw video
        if return_tensor:
            return to_show
        return to_show.numpy()

    def _encode_init_image(
        self,
        init_image: object,
        width: int,
        height: int,
        device: str,
        log_cb: LogCB = None,
    ) -> torch.Tensor:
        """Encode an input image into the Wan VAE latent space.

        The TurboDiffusion repo may expose different encoder helpers depending
        on version. We try a small set of common method names on the VAE
        interface and normalize the output to a 5D latent: (B,C,T,H,W).
        """

        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover - numpy is expected in runtime
            raise RuntimeError("numpy is required for init_image preprocessing") from exc

        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover - pillow should ship with gradio
            raise RuntimeError("Pillow is required for init_image preprocessing") from exc

        img = init_image
        if isinstance(img, (str, Path)):
            img = Image.open(img)

        if isinstance(img, Image.Image):
            pil = img.convert("RGB")
        elif isinstance(img, np.ndarray):
            arr = img
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.ndim != 3:
                raise ValueError(f"Unsupported init_image ndarray shape: {arr.shape}")
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.dtype != np.uint8:
                # Gradio may return float images in [0,1]
                arr_max = float(arr.max()) if arr.size else 0.0
                if arr_max <= 1.0:
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(arr, mode="RGB")
        else:
            raise TypeError(f"Unsupported init_image type: {type(init_image)!r}")

        # center-crop to target aspect ratio, then resize
        target_ratio = float(width) / float(height)
        cur_w, cur_h = pil.size
        cur_ratio = float(cur_w) / float(cur_h) if cur_h else target_ratio
        if cur_ratio > target_ratio:
            # too wide
            new_w = int(round(cur_h * target_ratio))
            left = max(0, (cur_w - new_w) // 2)
            pil = pil.crop((left, 0, left + new_w, cur_h))
        elif cur_ratio < target_ratio:
            # too tall
            new_h = int(round(cur_w / target_ratio)) if target_ratio else cur_h
            top = max(0, (cur_h - new_h) // 2)
            pil = pil.crop((0, top, cur_w, top + new_h))

        pil = pil.resize((int(width), int(height)), resample=Image.LANCZOS)
        arr = np.asarray(pil).astype(np.float32) / 255.0
        # (H,W,3) -> (1,3,H,W) in [-1,1]
        img_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        img_t = img_t * 2.0 - 1.0
        img_t = img_t.to(device=device, dtype=torch.float32)

        if log_cb:
            log_cb(f"[Engine] Init image prepared: shape={tuple(img_t.shape)} target={width}x{height}")

        latent = None
        for method_name in ("encode", "encode_image", "encode_images", "encode_video"):
            fn = getattr(self.tokenizer, method_name, None)
            if fn is None:
                continue
            try:
                latent = fn(img_t)
            except TypeError:
                # Some versions may require keyword args, but we keep it minimal.
                continue
            if latent is not None:
                break

        if latent is None:
            raise AttributeError(
                "VAE interface does not expose an image encoder; tried: encode/encode_image/encode_images/encode_video"
            )

        if isinstance(latent, (tuple, list)) and latent:
            latent = latent[0]

        if not isinstance(latent, torch.Tensor):
            raise TypeError(f"VAE encode returned unsupported type: {type(latent)!r}")

        # normalize latent shape to (B,C,T,H,W)
        if latent.ndim == 3:
            latent = latent.unsqueeze(0).unsqueeze(2)
        elif latent.ndim == 4:
            latent = latent.unsqueeze(2)
        elif latent.ndim == 5:
            pass
        else:
            raise ValueError(f"Unexpected latent ndim={latent.ndim} shape={tuple(latent.shape)}")

        # keep only first latent frame (I2V first-frame conditioning)
        latent = latent[:, :, 0:1, :, :]
        return latent.to(device=device)
