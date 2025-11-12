# ===============================
# NVIDIA CUDA base + PyTorch 2.0.1 + CUDA 11.8
# ===============================
FROM nvcr.io/nvidia/pytorch:23.05-py3

# Set up environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="All"

WORKDIR /workspace

# Optional: upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install core dependencies for NeMo
RUN pip install --no-cache-dir "nemo_toolkit[asr]==1.23.1" \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Optional tools for training / fine-tuning
RUN pip install pytorch-lightning==2.1.0 wandb matplotlib pandas

# Cleanup
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Default command
CMD ["bash"]
