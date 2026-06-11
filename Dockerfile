# ──────────────────────────────────────────────────────────────────────────────
# Base : Python 3.11 slim ARM64
# Compatible Raspberry Pi 5 (aarch64) et Jetson Orin (aarch64 + JetPack)
#
# Pour Jetson Orin : remplacer la base par
#   FROM nvcr.io/nvidia/l4t-pytorch:r36.x.x-pth2.x-py3
# et supprimer le bloc d'install PyTorch CPU ci-dessous.
# Le reste du Dockerfile reste identique.
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Variables build
ARG DEBIAN_FRONTEND=noninteractive

# ── Dépendances système ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        # audio
        libsndfile1 \
        ffmpeg \
        # build (requis pour certains wheels C)
        gcc \
        g++ \
        git \
        # nettoyage
    && rm -rf /var/lib/apt/lists/*

# ── Répertoires ───────────────────────────────────────────────────────────────
WORKDIR /app
RUN mkdir -p /data /output /root/.cache/torch/hub

# ── PyTorch CPU (ARM64 générique) ─────────────────────────────────────────────
# NOTE Jetson Orin : cette ligne devient inutile si vous partez de l4t-pytorch.
# On installe la version CPU pour éviter de tirer cuda+cudnn inutilement sur Pi.
RUN pip install --no-cache-dir \
        torch==2.3.1 \
        torchaudio==2.3.1 \
        --index-url https://download.pytorch.org/whl/cpu

# ── Dépendances Python ────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Code applicatif ───────────────────────────────────────────────────────────
COPY process.py .

# ── Cache torch.hub persistant (Silero) ──────────────────────────────────────
# Le volume /root/.cache/torch/hub peut être monté pour éviter de re-télécharger
# Silero VAD à chaque run.
ENV TORCH_HOME=/root/.cache/torch

# ── Point d'entrée ────────────────────────────────────────────────────────────
ENTRYPOINT ["python", "process.py"]
# Exemple d'utilisation :
#   docker run --rm \
#     --env-file .env \
#     -v $(pwd)/audio:/data \
#     -v $(pwd)/output:/output \
#     audio-diarization /data/mon_audio.wav --output /output/result.json
