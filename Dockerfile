# ──────────────────────────────────────────────────────────────────────────────
# PA-10 — Pipeline Audio Complète
# Cible : Raspberry Pi 5 (ARM64, CPU-only)
#
# Migration Jetson Orin :
#   1. Remplacer la base image par :
#        FROM nvcr.io/nvidia/l4t-pytorch:r36.x.x-pth2.x-py3
#   2. Supprimer le bloc "Install PyTorch CPU" ci-dessous
#      (PyTorch est déjà inclus dans l4t-pytorch)
#   3. Dans process.py, le device "cuda" sera automatiquement
#      sélectionné si torch.cuda.is_available() retourne True
#   4. Ajouter --runtime nvidia au docker run
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm

# ── Dépendances système ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install PyTorch CPU ───────────────────────────────────────────────────────
# Bloc à SUPPRIMER pour la migration Jetson (PyTorch déjà dans l4t-pytorch)
RUN pip install --no-cache-dir \
    torch==2.3.1 \
    torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cpu

# ── Dépendances Python ────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Code applicatif ───────────────────────────────────────────────────────────
COPY process.py .

# ── Dossiers de données ───────────────────────────────────────────────────────
RUN mkdir -p /data /output /workdir

# ── Cache modèles (persisté via volumes Docker) ───────────────────────────────
ENV TORCH_HOME=/cache/torch
ENV HF_HOME=/cache/huggingface

# ── Entrée ────────────────────────────────────────────────────────────────────
ENTRYPOINT ["python", "process.py"]
CMD ["--help"]
