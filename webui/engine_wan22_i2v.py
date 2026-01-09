import math
from pathlib import Path
from typing import Callable, Optional, Union

import torch
from einops import rearrange, repeat

from imaginaire.utils.io import save_image_or_video
from imaginaire.utils import log

from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface

from turbodiffusion.inference.modify_model import tensor_kwargs, create_model


ProgressCB = Optional[Callable[[str, int, int], None]]
LogCB = Optional[Callable[[str], None]]


class TurboWanI2VEngine:
    """
    Wan2.2 I2V engine (high-noise + low-noise models), matching
    ``turbodiffusion/inference/wan2.2_i2v_infer.py``.

    Key differences vs T2V:
      - requires two DiT checkpoints (high/low noise)
      - image is encoded into VAE latents to build y-conditioning:
          y = concat(mask(4ch), encoded_latents(16ch)) -> 20ch
      - net uses x (16ch) + y (20ch) internally (in_dim=36)
      - supports ODE sampling and adaptive resolution
    """

    def __init__(
        self,
        low_noise_model_path: str,
        high_noise_model_path: str,
        vae_path: str = "checkpoints/Wan2.1_VAE.pth",
        text_encoder_path: str = "checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
        model: str = "Wan2.2-A14B",
        resolution: str = "720p",
        aspect_ratio: str = "16:9",
        sigma_max: float = 200.0,
        boundary: float = 0.9,
        attention_type: str = "sagesla",
        sla_topk: float = 0.1,
        quant_linear: bool = False,
        default_norm: bool = False,
        keep_text_encoder: bool = False,
    ):
        try:
            torch._dynamo.config.suppress_errors = True
        except Exception:
            pass

        self.low_noise_model_path = low_noise_model_path
        self.high_noise_model_path = high_noise_model_path
        self.vae_path = vae_path
        self.text_encoder_path = text_encoder_path

        self.model = model
        self.resolution = resolution
        self.aspect_ratio = aspect_ratio
        self.sigma_max = sigma_max
        self.boundary = boundary
        self.attention_type = attention_type
        self.sla_topk = sla_topk
        self.quant_linear = quant_linear
        self.default_norm = default_norm
        self.keep_text_encoder = keep_text_encoder

        log.info(f"[Engine] Loading VAE: {vae_path}")
        self.tokenizer = Wan2pt1VAEInterface(vae_pth=vae_path)
        log.success("[Engine] VAE loaded.")

        self.args = self._make_args_namespace()

        log.info(f"[Engine] Loading DiT (high-noise): {high_noise_model_path}")
        self.high_noise_model = create_model(dit_path=high_noise_model_path, args=self.args).cpu().eval()
        torch.cuda.empty_cache()
        log.info(f"[Engine] Loading DiT (low-noise): {low_noise_model_path}")
        self.low_noise_model = create_model(dit_path=low_noise_model_path, args=self.args).cpu().eval()
        torch.cuda.empty_cache()
        log.success("[Engine] DiT models loaded.")

    def _make_args_namespace(self):
        class Args:
            pass

        args = Args()
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

    def _as_pil(self, init_image: object):
        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Pillow is required for I2V init image preprocessing") from exc

        if isinstance(init_image, Image.Image):
            return init_image.convert("RGB")

        if isinstance(init_image, (str, Path)):
            return Image.open(init_image).convert("RGB")

        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("numpy is required for I2V init image preprocessing") from exc

        if isinstance(init_image, np.ndarray):
            arr = init_image
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.ndim != 3:
                raise ValueError(f"Unsupported init_image ndarray shape: {arr.shape}")
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.dtype != np.uint8:
                arr_max = float(arr.max()) if arr.size else 0.0
                if arr_max <= 1.0:
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)
            return Image.fromarray(arr, mode="RGB")

        raise TypeError(f"Unsupported init_image type: {type(init_image)!r}")

    def _compute_target_size(
        self,
        pil_image,
        resolution: str,
        aspect_ratio: str,
        adaptive_resolution: bool,
        log_cb: LogCB = None,
    ) -> tuple[int, int]:
        if not adaptive_resolution:
            w, h = VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]
            return int(w), int(h)

        base_w, base_h = VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]
        max_area = int(base_w) * int(base_h)
        orig_w, orig_h = pil_image.size
        if orig_w <= 0 or orig_h <= 0:
            raise ValueError(f"Invalid init image size: {orig_w}x{orig_h}")

        image_aspect_ratio = float(orig_h) / float(orig_w)
        ideal_w = math.sqrt(max_area / image_aspect_ratio)
        ideal_h = math.sqrt(max_area * image_aspect_ratio)

        stride = int(self.tokenizer.spatial_compression_factor) * 2
        lat_h = max(1, round(ideal_h / stride))
        lat_w = max(1, round(ideal_w / stride))
        h = int(lat_h * stride)
        w = int(lat_w * stride)

        if log_cb:
            log_cb(f"[Engine] Adaptive resolution: input_ar={image_aspect_ratio:.4f} -> {w}x{h}")
        return w, h

    def _preprocess_image_tensor(self, pil_image, w: int, h: int) -> torch.Tensor:
        import numpy as np
        from PIL import Image

        pil_image = pil_image.resize((int(w), int(h)), resample=Image.BICUBIC)
        arr = np.asarray(pil_image).astype(np.float32) / 255.0
        img_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
        img_t = (img_t - 0.5) / 0.5  # [-1, 1]
        return img_t.to(device=tensor_kwargs["device"], dtype=torch.float32)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        init_image: object,
        num_steps: int = 4,
        num_frames: int = 81,
        seed: int = 0,
        num_samples: int = 1,
        resolution: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        adaptive_resolution: bool = True,
        boundary: Optional[float] = None,
        ode: bool = False,
        sigma_max: Optional[float] = None,
        save_path: Optional[Union[str, Path]] = None,
        fps: int = 16,
        return_tensor: bool = False,
        progress_cb: ProgressCB = None,
        log_cb: LogCB = None,
        use_tqdm: bool = False,
    ):
        if resolution is None:
            resolution = self.resolution
        if aspect_ratio is None:
            aspect_ratio = self.aspect_ratio
        if boundary is None:
            boundary = self.boundary
        if sigma_max is None:
            sigma_max = self.sigma_max

        self._update_args(
            prompt=prompt,
            num_steps=num_steps,
            num_frames=num_frames,
            seed=seed,
            num_samples=num_samples,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            sigma_max=sigma_max,
        )

        if log_cb:
            log_cb(f"[Engine] Embedding prompt: {prompt}")
        if progress_cb:
            progress_cb("embedding", 0, 1)

        text_emb = get_umt5_embedding(checkpoint_path=self.text_encoder_path, prompts=prompt).to(**tensor_kwargs)
        if not self.keep_text_encoder:
            clear_umt5_memory()

        if progress_cb:
            progress_cb("embedding", 1, 1)

        if log_cb:
            log_cb("[Engine] Loading and preprocessing init image...")
        if progress_cb:
            progress_cb("encode_image", 0, 1)

        pil_image = self._as_pil(init_image)
        w, h = self._compute_target_size(pil_image, resolution, aspect_ratio, adaptive_resolution, log_cb=log_cb)

        F = int(num_frames)
        lat_h = int(h) // int(self.tokenizer.spatial_compression_factor)
        lat_w = int(w) // int(self.tokenizer.spatial_compression_factor)
        lat_t = int(self.tokenizer.get_latent_num_frames(F))

        image_tensor = self._preprocess_image_tensor(pil_image, w=w, h=h)

        frames = torch.zeros(
            1,
            3,
            F,
            int(h),
            int(w),
            device=image_tensor.device,
            dtype=image_tensor.dtype,
        )
        frames[:, :, 0, :, :] = image_tensor
        encoded_latents = self.tokenizer.encode(frames)  # (B,C_lat,T_lat,H_lat,W_lat)
        del frames
        torch.cuda.empty_cache()

        msk = torch.zeros(
            1,
            4,
            lat_t,
            lat_h,
            lat_w,
            device=tensor_kwargs["device"],
            dtype=tensor_kwargs["dtype"],
        )
        msk[:, :, 0, :, :] = 1.0

        y = torch.cat([msk, encoded_latents.to(**tensor_kwargs)], dim=1)
        y = y.repeat(int(num_samples), 1, 1, 1, 1)

        condition = {
            "crossattn_emb": repeat(text_emb.to(**tensor_kwargs), "b l d -> (k b) l d", k=int(num_samples)),
            "y_B_C_T_H_W": y,
        }

        if progress_cb:
            progress_cb("encode_image", 1, 1)

        if log_cb:
            log_cb("[Engine] Preparing noise + timesteps...")
        if progress_cb:
            progress_cb("setup", 0, 1)

        state_shape = [self.tokenizer.latent_ch, lat_t, lat_h, lat_w]

        generator = torch.Generator(device=tensor_kwargs["device"])
        generator.manual_seed(int(seed))

        init_noise = torch.randn(
            int(num_samples),
            *state_shape,
            dtype=torch.float32,
            device=tensor_kwargs["device"],
            generator=generator,
        )

        mid_t = [1.5, 1.4, 1.0][: max(int(num_steps) - 1, 0)]
        t_steps = torch.tensor(
            [math.atan(float(sigma_max)), *mid_t, 0],
            dtype=torch.float64,
            device=init_noise.device,
        )
        t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

        if progress_cb:
            progress_cb("setup", 1, 1)

        if log_cb:
            log_cb(f"[Engine] Sampling (high→low @ boundary={boundary}, ode={ode})")
        total_steps = t_steps.shape[0] - 1
        if progress_cb:
            progress_cb("sampling", 0, total_steps)

        x = init_noise.to(torch.float64) * t_steps[0]
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)

        self.high_noise_model.cuda()
        net = self.high_noise_model
        switched = False

        pairs = list(zip(t_steps[:-1], t_steps[1:]))
        it = pairs
        if use_tqdm:
            from tqdm import tqdm

            it = tqdm(pairs, desc="Sampling", total=total_steps)

        for i, (t_cur, t_next) in enumerate(it):
            if float(t_cur.item()) < float(boundary) and not switched:
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()
                self.low_noise_model.cuda()
                net = self.low_noise_model
                switched = True
                if log_cb:
                    log_cb("[Engine] Switched to low-noise model.")

            v_pred = net(
                x_B_C_T_H_W=x.to(**tensor_kwargs),
                timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
                **condition,
            ).to(torch.float64)

            if ode:
                x = x - (t_cur - t_next) * v_pred
            else:
                x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                    *x.shape,
                    dtype=torch.float32,
                    device=tensor_kwargs["device"],
                    generator=generator,
                )

            if progress_cb:
                progress_cb("sampling", i + 1, total_steps)

        samples = x.float()
        if switched:
            self.low_noise_model.cpu()
        else:
            self.high_noise_model.cpu()
        torch.cuda.empty_cache()

        if log_cb:
            log_cb("[Engine] Decoding VAE...")
        if progress_cb:
            progress_cb("decode", 0, 1)

        video = self.tokenizer.decode(samples)  # (B,C,T,H,W)
        to_show = (1.0 + video.clamp(-1, 1)) / 2.0
        to_show = to_show.float().cpu()

        if progress_cb:
            progress_cb("decode", 1, 1)

        if save_path is not None:
            if log_cb:
                log_cb(f"[Engine] Saving video: {save_path}")
            if progress_cb:
                progress_cb("save", 0, 1)

            save_path = str(save_path)
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            # match upstream tiling: (C,T,(n*h),(b*w))
            save_image_or_video(rearrange(to_show.unsqueeze(0), "n b c t h w -> c t (n h) (b w)"), save_path, fps=int(fps))

            if progress_cb:
                progress_cb("save", 1, 1)
            return save_path

        if return_tensor:
            return to_show
        return to_show.numpy()
