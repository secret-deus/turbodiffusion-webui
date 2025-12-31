# 推荐使用 NVIDIA 官方 CUDA 镜像（因为 pytorch/pytorch:2.8.0-cuda12.1-cudnn9-devel 你遇到过 tag not found）
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates \
    build-essential ninja-build cmake pkg-config \
    ffmpeg libglib2.0-0 libsm6 libxrender1 libxext6 \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# python
RUN python3 -m pip install --upgrade pip setuptools wheel

# Clone TurboDiffusion
RUN git clone https://github.com/thu-ml/TurboDiffusion.git
WORKDIR /workspace/TurboDiffusion

# Install torch (CUDA 12.8 需要从官方 index 拉)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install deps
RUN pip install --no-cache-dir gradio==6.0.1 einops tqdm safetensors
RUN pip install --no-cache-dir -r requirements.txt

# Optional: enable sagesla (SpargeAttn) ——不装也能跑 sla/original
# RUN pip install --no-cache-dir "git+https://github.com/thu-ml/SpargeAttn.git" --no-build-isolation

# Download checkpoints (官方要求)
RUN mkdir -p checkpoints && cd checkpoints && \
    wget -q --show-progress -O Wan2.1_VAE.pth \
      https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_VAE.pth && \
    wget -q --show-progress -O models_t5_umt5-xxl-enc-bf16.pth \
      https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth

# Download TurboDiffusion distilled DiT checkpoints
# 1.3B 480P quant
RUN cd checkpoints && \
    wget -q --show-progress -O TurboWan2.1-T2V-1.3B-480P-quant.pth \
      "https://huggingface.co/TurboDiffusion/TurboWan2.1-T2V-1.3B-480P/resolve/main/TurboWan2.1-T2V-1.3B-480P-quant.pth?download=true"

# 14B 720P quant + fp16
RUN cd checkpoints && \
    wget -q --show-progress -O TurboWan2.1-T2V-14B-720P-quant.pth \
      "https://huggingface.co/TurboDiffusion/TurboWan2.1-T2V-14B-720P/resolve/main/TurboWan2.1-T2V-14B-720P-quant.pth?download=true" && \
    wget -q --show-progress -O TurboWan2.1-T2V-14B-720P.pth \
      "https://huggingface.co/TurboDiffusion/TurboWan2.1-T2V-14B-720P/resolve/main/TurboWan2.1-T2V-14B-720P.pth?download=true"

# Copy our WebUI code (你可以把下面文件放在 repo 根目录，然后 docker build 会把它们复制进去)
COPY app_gradio.py /workspace/TurboDiffusion/app_gradio.py
COPY webui /workspace/TurboDiffusion/webui

EXPOSE 7860
CMD ["python3", "app_gradio.py"]
