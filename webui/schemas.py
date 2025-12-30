from dataclasses import dataclass

@dataclass(frozen=True)
class EngineConfig:
    name: str
    dit_path: str
    vae_path: str
    text_encoder_path: str
    model: str = "Wan2.1-1.3B"
    resolution: str = "480p"
    aspect_ratio: str = "16:9"
    quant_linear: bool = True
    default_norm: bool = False

PRESETS = {
    "Wan2.1 T2V 1.3B 480p (quant)": EngineConfig(
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

    "Wan2.1 T2V 14B 720p (quant, 5090 recommended)": EngineConfig(
        name="Wan2.1 T2V 14B 720p (quant, 5090 recommended)",
        dit_path="checkpoints/TurboWan2.1-T2V-14B-720P-quant.pth",
        vae_path="checkpoints/Wan2.1_VAE.pth",
        text_encoder_path="checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
        model="Wan2.1-14B",
        resolution="720p",
        aspect_ratio="16:9",
        quant_linear=True,
        default_norm=False,
    ),

    "Wan2.1 T2V 14B 720p (fp16, >40GB GPU)": EngineConfig(
        name="Wan2.1 T2V 14B 720p (fp16, >40GB GPU)",
        dit_path="checkpoints/TurboWan2.1-T2V-14B-720P.pth",
        vae_path="checkpoints/Wan2.1_VAE.pth",
        text_encoder_path="checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
        model="Wan2.1-14B",
        resolution="720p",
        aspect_ratio="16:9",
        quant_linear=False,
        default_norm=False,
    ),
}