# Audio Diarization — Docker (Raspberry Pi 5 / Jetson Orin)

Pipeline de diarisation vocale : **Silero VAD** + **pyannote.audio 3.1**  
Entrée : fichier audio (wav, mp3, flac, ogg…)  
Sortie : JSON avec timestamps par locuteur

---

## Prérequis

- Docker ≥ 24 (testé avec 29.5.3)
- Raspberry Pi 5 — 8 Go RAM recommandés (pyannote est gourmand)
- Token HuggingFace avec accès à `pyannote/speaker-diarization-3.1`
  → Accepter la licence sur : https://huggingface.co/pyannote/speaker-diarization-3.1

---

## Installation

```bash
# 1. Cloner / copier le projet
cd audio-diarization

# 2. Créer le fichier .env
cp .env.example .env
# Éditer .env et renseigner votre HF_TOKEN

# 3. Créer les dossiers de travail
mkdir -p audio output

# 4. Build de l'image (15-20 min sur Pi 5, télécharge ~3 Go)
docker build -t audio-diarization .
```

---

## Utilisation

### Traitement d'un fichier (résultat sur stdout)

```bash
docker run --rm \
  --env-file .env \
  -v $(pwd)/audio:/data:ro \
  audio-diarization /data/mon_audio.wav
```

### Résultat dans un fichier JSON

```bash
docker run --rm \
  --env-file .env \
  -v $(pwd)/audio:/data:ro \
  -v $(pwd)/output:/output \
  audio-diarization /data/mon_audio.wav --output /output/result.json
```

### Avec docker compose (recommandé)

```bash
# Copier l'audio
cp mon_audio.wav audio/

# Lancer le traitement
docker compose run --rm diarization /data/mon_audio.wav --output /output/result.json

# Lire le résultat
cat output/result.json
```

---

## Format de sortie JSON

```json
{
  "metadata": {
    "audio_file": "mon_audio.wav",
    "duration_s": 42.5,
    "processed_at": "2025-06-10T14:32:00",
    "num_speakers_detected": 2,
    "num_vad_segments": 12
  },
  "segments": [
    { "start": 0.512, "end": 3.840, "speaker": "SPEAKER_00" },
    { "start": 4.200, "end": 7.120, "speaker": "SPEAKER_01" }
  ]
}
```

---

## Performance sur Raspberry Pi 5

| Durée audio | Temps de traitement (CPU) |
|-------------|--------------------------|
| 1 min       | ~3-5 min                 |
| 5 min       | ~15-25 min               |
| 15 min      | ~45-75 min               |

pyannote tourne entièrement en CPU sur Pi — c'est normal, prévoir du temps.

---

## Migration vers Jetson Orin

Deux changements à faire dans le `Dockerfile` :

```dockerfile
# 1. Remplacer la base image
FROM nvcr.io/nvidia/l4t-pytorch:r36.x.x-pth2.x-py3

# 2. Supprimer ce bloc (PyTorch est déjà dans l4t-pytorch)
# RUN pip install torch torchaudio --index-url .../cpu
```

Et dans `process.py`, activer CUDA :

```python
# Ligne ~57 — remplacer cpu par cuda
pipeline = pipeline.to(torch.device("cuda"))
```

Sur Jetson, ajouter `--runtime nvidia` au `docker run` :

```bash
docker run --rm --runtime nvidia \
  --env-file .env \
  -v $(pwd)/audio:/data:ro \
  -v $(pwd)/output:/output \
  audio-diarization /data/mon_audio.wav --output /output/result.json
```

---

## Cache des modèles

Les volumes Docker `torch_hub_cache` et `hf_cache` persistent le cache des modèles entre les runs.  
Silero VAD (~3 Mo) et pyannote (~300 Mo) ne sont téléchargés qu'une seule fois.

Pour forcer un re-téléchargement :

```bash
docker volume rm audio-diarization_torch_hub_cache audio-diarization_hf_cache
```
