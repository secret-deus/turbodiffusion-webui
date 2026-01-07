FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=12.0

WORKDIR /workspace

# system deps (根据 history：只装了这些)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates \
    python3.10-dev python3-dev \
    ffmpeg \
    build-essential ninja-build cmake \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*


# clone repo
RUN git clone https://github.com/thu-ml/TurboDiffusion.git
WORKDIR /workspace/TurboDiffusion

# submodules
RUN git submodule update --init --recursive

# upgrade pip toolchain
RUN pip install -U pip setuptools wheel

# install turbodiffusion (你历史里就是这么装的)
RUN pip install -e . --no-build-isolation

# install gradio（你历史是 pip install -U gradio）
RUN pip install -U gradio

# checkpoints
ENV CKPT_DIR=/workspace/TurboDiffusion/checkpoints
RUN mkdir -p ${CKPT_DIR}

# 用 huggingface_hub 下载 TurboWan 1.3B 480P quant（history 里就是这种方式）
RUN python - <<'PY'
from huggingface_hub import hf_hub_download
repo_id = "TurboDiffusion/TurboWan2.1-T2V-14B-720P"
filename = "TurboWan2.1-T2V-14B-720P-quant.pth"
path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir="checkpoints", local_dir_use_symlinks=False)
print("Downloaded:", path)
PY

RUN set -eux; \
    CKPT_DIR=/workspace/TurboDiffusion/checkpoints; \
    mkdir -p "$CKPT_DIR"; \
    ls -la /workspace/TurboDiffusion; \
    wget -L --progress=dot:giga -O "$CKPT_DIR/Wan2.1_VAE.pth" \
      https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_VAE.pth; \
    wget -L --progress=dot:giga -O "$CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
      https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth

# 复制 WebUI
COPY app_gradio.py /workspace/TurboDiffusion/turbodiffusion/app_gradio.py
COPY webui/ /workspace/TurboDiffusion/turbodiffusion/webui/

ENV HF_HOME=/workspace/hf_cache
ENV TRANSFORMERS_CACHE=/workspace/hf_cache
ENV HF_HUB_DISABLE_TELEMETRY=1

COPY hf_models/google/umt5-xxl/ /workspace/hf_cache/hub/models--google--umt5-xxl

# 关键：固定 PYTHONPATH（这就是你之前解决 rcm 的手段）
ENV PYTHONPATH=/workspace/TurboDiffusion/turbodiffusion


EXPOSE 7860

# 你 history 最顶层的命令是 bash -lc python app_gradio.py
CMD ["bash", "-lc", "python turbodiffusion/app_gradio.py"]