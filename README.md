# PA-10 — Pipeline Audio Complète (Docker — Raspberry Pi 5 / Jetson Orin)

Pipeline de traitement audio : **Silero VAD** + **pyannote.audio 3.1** + **faster-whisper**  
Entrée : fichier audio (wav, mp3, flac, ogg…)  
Sortie : JSON avec timestamps par locuteur, wake words détectés, et mapping conducteur/passager

---

## Modules intégrés

| Module | Modèle | Description |
|--------|--------|-------------|
| VAD | Silero VAD | Détection des segments de parole |
| Diarisation | pyannote/speaker-diarization-3.1 | Attribution par locuteur |
| Transcription | faster-whisper base INT8 | Transcription CPU-optimisée |
| Wake word | — | Détection "Hey Assistant" |
| Enrôlement | pyannote/embedding | Identification conducteur/passager |

---

## Prérequis

- Docker ≥ 24 (testé avec 29.5.3)
- Raspberry Pi 5 — 8 Go RAM recommandés
- Token HuggingFace avec accès aux modèles pyannote :
  - https://huggingface.co/pyannote/speaker-diarization-3.1
  - https://huggingface.co/pyannote/embedding

---

## Installation

```bash
# 1. Cloner / copier le projet
cd pipeline-complete

# 2. Créer le fichier .env
cp .env.example .env
# Éditer .env et renseigner votre HF_TOKEN

# 3. Créer les dossiers de travail
mkdir -p audio output

# 4. Build de l'image (20-30 min sur Pi 5 — télécharge PyTorch + pyannote + Whisper)
docker compose build
```

---

## Utilisation

### Traitement simple (résultat sur stdout)

```bash
docker compose run --rm pipeline-complete /data/mon_audio.wav
```

### Résultat dans un fichier JSON

```bash
docker compose run --rm pipeline-complete /data/mon_audio.wav --output /output/result.json
```

### Avec fichier d'enrôlement séparé (conducteur/passager)

Préparez un fichier audio d'enrôlement avec ce script :
```
[silence 2s]
[PERSONNE 1] "C'est moi, je suis le conducteur."
[silence 2s]
[PERSONNE 2] "C'est moi, je suis le passager."
```

```bash
cp enrol.wav audio/
docker compose run --rm pipeline-complete /data/mon_audio.wav \
  --enroll /data/enrol.wav \
  --output /output/result.json
```

### Options disponibles

```
positional arguments:
  audio                 Fichier audio à traiter

options:
  --output, -o          Chemin du JSON de sortie (défaut : stdout)
  --enroll              Fichier audio d'enrôlement conducteur/passager
  --num-speakers N      Nombre de locuteurs attendu (améliore la diarisation)
  --language LANG       Langue pour Whisper (défaut : fr)
  --whisper-model SIZE  Taille du modèle Whisper : tiny / base / small (défaut : base)
```

---

## Format de sortie JSON

```json
{
  "metadata": {
    "audio_file": "mon_audio.wav",
    "duration_s": 42.5,
    "processed_at": "2025-06-10T14:32:00",
    "vad_model": "silero-vad",
    "diarization_model": "pyannote/speaker-diarization-3.1",
    "transcription_model": "faster-whisper-base-int8",
    "speakers_detected": ["SPEAKER_00", "SPEAKER_01"],
    "num_speakers_detected": 2,
    "num_vad_segments": 12
  },
  "diarization_segments": [
    { "start": 0.512, "end": 3.840, "speaker": "SPEAKER_00" },
    { "start": 4.200, "end": 7.120, "speaker": "SPEAKER_01" }
  ],
  "wake_word_triggers": [
    {
      "segment_id": 3,
      "timestamp_start": 8.100,
      "timestamp_end": 9.400,
      "trigger_word": "hey assistant",
      "transcript": "hey assistant allume la radio",
      "active_speaker": "SPEAKER_00"
    }
  ],
  "role_mapping": {
    "SPEAKER_00": "conducteur",
    "SPEAKER_01": "passager"
  },
  "der_score": null
}
```

---

## Performance sur Raspberry Pi 5

| Durée audio | Temps estimé (CPU) |
|-------------|-------------------|
| 1 min | ~5-10 min |
| 5 min | ~25-45 min |
| 15 min | ~75-130 min |

La pipeline est plus lente que la diarisation seule car elle ajoute la transcription Whisper sur chaque segment VAD.  
Pour réduire le temps : `--whisper-model tiny` (moins précis, ~2× plus rapide).

---

## Cache des modèles

Les volumes Docker `torch_hub_cache` et `hf_cache` persistent le cache entre les runs.  
Silero VAD (~3 Mo), pyannote (~300 Mo), et Whisper base (~150 Mo) ne sont téléchargés qu'une seule fois.

Pour forcer un re-téléchargement :

```bash
docker volume rm pa10-diarization_torch_hub_cache pa10-diarization_hf_cache
```

---

## Migration vers Jetson Orin

Deux changements dans le `Dockerfile` :

```dockerfile
# 1. Remplacer la base image
FROM nvcr.io/nvidia/l4t-pytorch:r36.x.x-pth2.x-py3

# 2. Supprimer le bloc "Install PyTorch CPU"
# (PyTorch est déjà inclus dans l4t-pytorch)
```

Dans `docker-compose.yml`, décommenter le bloc `deploy` :

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

Sur Jetson, ajouter `--runtime nvidia` si vous utilisez `docker run` directement :

```bash
docker run --rm --runtime nvidia \
  --env-file .env \
  -v $(pwd)/audio:/data:ro \
  -v $(pwd)/output:/output \
  pa10-diarization /data/mon_audio.wav --output /output/result.json
```

Le device CUDA est détecté automatiquement dans `process.py` via `torch.cuda.is_available()`.  
Aucune modification du code Python n'est nécessaire.

---

## Intégration DER (à venir)

La section DER (`der_score` dans le JSON) est prévue pour l'intégration du code Ranim (PA-20).  
Structure attendue pour le calcul via `pyannote.metrics` :

```python
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

reference = Annotation()
# Remplir avec les annotations RTTM de référence

metric = DiarizationErrorRate()
der = metric(reference, hypothesis)
```
