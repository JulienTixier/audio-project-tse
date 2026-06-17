# PA-10 — Pipeline Audio Complète (Docker — Raspberry Pi 5 / Jetson Orin)

Pipeline de traitement audio : **Silero VAD** + **pyannote.audio 3.1** + **faster-whisper** + **audeering wav2vec2**  
Entrée : fichier audio (wav, mp3, flac, ogg…)  
Sortie : JSON avec timestamps par locuteur, wake words, mapping conducteur/passager, genre/âge, et DER optionnel

---

## Modules intégrés

| Module | Modèle | Description |
|--------|--------|-------------|
| VAD | Silero VAD | Détection des segments de parole |
| Diarisation | pyannote/speaker-diarization-3.1 | Attribution par locuteur |
| Embedding | pyannote/embedding | Empreinte vocale pour l'enrôlement |
| Transcription | faster-whisper base INT8 | Transcription CPU-optimisée |
| Wake word | — | Détection "Hey Assistant" |
| Enrôlement | pyannote/embedding | Identification conducteur/passager |
| Genre + Âge | Pitch F0 + audeering/wav2vec2-large-robust-6-ft-age-gender | Genre par autocorrélation, âge par wav2vec2 |
| DER | pyannote.metrics | Calcul optionnel via fichier RTTM de référence |

*Détection d'émotion : non implémentée dans cette version.*

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
cp .env.example .env   # renseigner HF_TOKEN
mkdir -p audio output
docker compose build   # ~25-35 min sur Pi 5 (audeering ~1.2 Go supplémentaires)
```

---

## Utilisation

### Traitement simple

```bash
docker compose run --rm pipeline_complete /data/mon_audio.wav --output /output/result.json
```

### Avec nombre de locuteurs connu (recommandé)

```bash
docker compose run --rm pipeline_complete /data/mon_audio.wav \
  --num-speakers 2 \
  --output /output/result.json
```

### Avec fichier d'enrôlement séparé

```bash
docker compose run --rm pipeline_complete /data/mon_audio.wav \
  --enroll /data/enrol.wav \
  --output /output/result.json
```

### Avec calcul du DER (fichier RTTM de vérité terrain)

```bash
docker compose run --rm pipeline_complete /data/mon_audio.wav \
  --reference /data/verite_terrain.rttm \
  --output /output/result.json
```

### Options disponibles

```
positional:
  audio                 Fichier audio à traiter

options:
  --output, -o          Chemin du JSON de sortie (défaut : stdout)
  --enroll              Fichier audio d'enrôlement conducteur/passager
  --reference           Fichier RTTM de vérité terrain pour le DER (optionnel)
  --num-speakers N      Nombre de locuteurs attendu
  --language LANG       Langue pour Whisper (défaut : fr)
  --whisper-model SIZE  tiny / base / small (défaut : base)
```

---

## Format de sortie JSON

```json
{
  "metadata": {
    "audio_file": "audio.wav",
    "duration_s": 68.3,
    "genre_model": "pitch-F0-autocorrelation + audeering-wav2vec2",
    "modules_integres": ["VAD", "diarisation", "transcription", "wake_word", "conducteur_passager", "genre_age", "DER"]
  },
  "diarization_segments": [...],
  "wake_word_triggers": [...],
  "role_mapping": {
    "SPEAKER_00": "conducteur",
    "SPEAKER_01": "passager"
  },
  "genre_age_par_locuteur": {
    "SPEAKER_00": {
      "gender": "male",
      "confidence": 0.87,
      "pitch_f0_mean": 118.4,
      "age_estimate": 34.2,
      "role": "conducteur",
      "total_speech_s": 22.1
    }
  },
  "der_score": 18.4
}
```

---

## Performance sur Raspberry Pi 5

Le modèle audeering (~1.2 Go) alourdit le build initial mais son inférence reste rapide (traitement par locuteur sur audio concaténé, pas segment par segment).

| Durée audio | Temps estimé (CPU) | Avec --whisper-model tiny |
|-------------|-------------------|--------------------------|
| 30 s | ~4-7 min | ~3-4 min |
| 1 min | ~10-15 min | ~6-9 min |
| 5 min | ~40-60 min | ~25-35 min |

---

## Cache des modèles

Les volumes `torch_hub_cache` et `hf_cache` persistent entre les runs.  
Téléchargements au premier run : Silero (~3 Mo), pyannote (~300 Mo), Whisper base (~150 Mo), audeering (~1.2 Go).

```bash
# Forcer un re-téléchargement
docker volume rm pa10-diarization_torch_hub_cache pa10-diarization_hf_cache
```

---

## Migration vers Jetson Orin

Dans le `Dockerfile`, remplacer la base image et supprimer le bloc PyTorch CPU :

```dockerfile
FROM nvcr.io/nvidia/l4t-pytorch:r36.x.x-pth2.x-py3
# Supprimer le bloc "Install PyTorch CPU"
```

Dans `docker-compose.yml`, décommenter le bloc `deploy` GPU.

Aucune modification de `process.py` — le device cuda/cpu est auto-détecté.
