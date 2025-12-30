# TurboDiffusion WebUI (Wan2.1 T2V Engine Mode) 🚀

A **production-friendly WebUI** for [thu-ml/TurboDiffusion](https://github.com/thu-ml/TurboDiffusion) using an **engine-based inference backend**.

✅ **Load model once** → ✅ **Reuse model for multiple generations** → ✅ **Docker Compose deployment** → ✅ **Wan2.1 T2V 1.3B / 14B presets** → ✅ **Gradio 6 compatible UI**

---

## ✨ Features

### ✅ Engine Mode (Core)
- **Model warm-load**: load DiT + VAE + umT5 encoder once at startup
- **Multi-run reuse**: subsequent generations reuse the model (no subprocess reload)
- **Stable GPU memory**: avoids repeated initialization spikes

### ✅ WebUI
- Switch between presets:
  - **Wan2.1 T2V 1.3B 480p quant**
  - **Wan2.1 T2V 14B 720p quant**
  - **Wan2.1 T2V 14B 720p fp16** (requires large VRAM)
- Useful knobs:
  - steps (1~4)
  - frames (17~81)
  - seed mode (fixed/random)
  - attention type (`sla / sagesla / original`)
  - `sla_topk`, `sigma_max`, `default_norm`
  - FPS
- Built-in logs & status display
- Output video is **read-only** (not uploadable)

### ✅ Docker-first
- All dependencies + model checkpoints are fetched in the Dockerfile
- One command launch with Docker Compose
- Works with NVIDIA Container Runtime (GPU)

---

## 🧠 Supported Models / Presets

| Preset | Resolution | VRAM Suggestion | Notes |
|------|------------|-----------------|------|
| Wan2.1 T2V **1.3B** (quant) | 480p | ~12GB+ | Safe for 2080Ti / 3090 / 4090 |
| Wan2.1 T2V **14B** (quant) | 720p | ~24GB+ | Recommended on 4090 / 5090 |
| Wan2.1 T2V **14B** (fp16) | 720p | **40GB+** | A100 40G / 80G etc |

> You mentioned you upgraded to **RTX 5090** — 14B quant 720p should be the sweet spot.

---

## 📦 Quick Start (Docker Compose)

### 1) Requirements
- Linux with NVIDIA GPU
- Docker >= 24
- Docker Compose v2
- NVIDIA Container Toolkit installed

Check:
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
````

### 2) Clone this repo / prepare files

In project root:

```
TurboDiffusion/
├── Dockerfile
├── docker-compose.yml
├── app_gradio.py
└── webui/
    ├── engine_wan21.py
    ├── manager.py
    ├── schemas.py
    ├── utils.py
    └── __init__.py
```

### 3) Launch

```bash
docker compose up --build
```

Open:

```
http://<server-ip>:7860
```

---

## 🐳 Dockerfile Notes (Models Included)

The Dockerfile will automatically download:

✅ **Wan2.1_VAE.pth**
✅ **models_t5_umt5-xxl-enc-bf16.pth**
✅ Turbo DiT distilled checkpoints:

* `TurboWan2.1-T2V-1.3B-480P-quant.pth`
* `TurboWan2.1-T2V-14B-720P-quant.pth`
* `TurboWan2.1-T2V-14B-720P.pth`

All are placed under:

```
/workspace/TurboDiffusion/checkpoints/
```

---

## ⚙️ WebUI Usage

### ✅ Generate Tab

1. Choose a model preset
2. Enter prompt
3. Choose steps / frames / seed
4. Click **Generate**
5. Output video is shown on the right

### ✅ Models Tab

* Validate checkpoints
* Load model
* Unload model
* GPU info & SpargeAttn check

---

## ⚡ SageSLA Support (Optional)

`attention_type="sagesla"` requires SpargeAttn.

If SpargeAttn is not installed:

* UI will auto hide `sagesla` option
* Engine will fallback `sagesla -> sla`

To enable:

```dockerfile
RUN pip install --no-cache-dir "git+https://github.com/thu-ml/SpargeAttn.git" --no-build-isolation
```

---

## 🧩 Add Your Own Model Preset

Edit:
`webui/schemas.py`

```python
PRESETS["MyModelName"] = EngineConfig(
    name="MyModelName",
    dit_path="checkpoints/xxx.pth",
    vae_path="checkpoints/Wan2.1_VAE.pth",
    text_encoder_path="checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
    model="Wan2.1-14B",
    resolution="720p",
    aspect_ratio="16:9",
    quant_linear=True,
    default_norm=False,
)
```

Then rebuild:

```bash
docker compose up --build
```

---

## 🛠 Troubleshooting

### 1) `torch.cuda.is_available() = False`

Inside container:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Ensure:

* docker run has `--gpus all`
* NVIDIA container runtime installed
* Driver supports CUDA version

### 2) `sagesla` assertion error

You are using `attention_type=sagesla` but missing SpargeAttn.

Solution:

* Install SpargeAttn in Dockerfile
* Or switch to `sla/original`

### 3) `--dit_path required`

This is correct: distilled models require DiT checkpoint path.

Presets already fill `dit_path`.
Do not run the raw script without providing `--dit_path`.

### 4) `imaginaire not installed`

This project uses a fallback video saving implementation.
No need to install `imaginaire`.

---

## 📂 Output Files

Outputs are mounted to host:

```
./outputs
```

---

## 🔥 Performance Tips (RTX 5090)

Recommended:

* **14B quant 720p**
* steps = 4
* frames = 81
* `attention_type = sla` or `sagesla` (if installed)
* keep container persistent (do not restart)

---

## 📌 Credits

* TurboDiffusion: [https://github.com/thu-ml/TurboDiffusion](https://github.com/thu-ml/TurboDiffusion)
* Wan2.1 checkpoints: [https://huggingface.co/Wan-AI](https://huggingface.co/Wan-AI)
* Distilled DiT checkpoints: TurboDiffusion official release