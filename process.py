#!/usr/bin/env python3
"""
PA-10 — Pipeline Complète de Traitement Audio
Sprint 5 | Équipe P07 | Projet Stellantis

Modules : VAD (Silero) + Diarisation (pyannote) + Transcription (faster-whisper)
          + Wake word + Enrôlement conducteur/passager + Export JSON

Usage:
    python process.py <audio_file> [--output <out.json>] [--enroll <enroll.wav>]
                      [--num-speakers <n>] [--language <lang>]
"""

import argparse
import datetime
import json
import logging
import os
import sys
import tempfile

import numpy as np
import soundfile as sf
import torch
import torchaudio
from pydub import AudioSegment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SAMPLE_RATE = 16000

WAKE_WORD_VARIANTS = [
    "hey assistant",
    "assistant",
    "hey assistan",
    "elles assistent",
    "et assistant",
    "hé assistant",
    "assiste ton",
]


# ─────────────────────────────────────────────────────────────────────────────
# Chargement des modèles
# ─────────────────────────────────────────────────────────────────────────────

def load_models(hf_token: str, whisper_model_size: str = "base", device: str = "cpu"):
    """Charge tous les modèles nécessaires à la pipeline."""

    # Patch compatibilité torchaudio / pyannote
    if not hasattr(torchaudio, "AudioMetaData"):
        try:
            from torchaudio._backend.utils import AudioMetaData
        except ImportError:
            import collections
            AudioMetaData = collections.namedtuple(
                "AudioMetaData",
                ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
            )
        torchaudio.AudioMetaData = AudioMetaData

    log.info("Chargement pyannote diarisation…")
    from pyannote.audio import Pipeline, Model, Inference

    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )
    diarization_pipeline = diarization_pipeline.to(torch.device(device))
    log.info("✅ pyannote diarisation chargé")

    log.info("Chargement pyannote embedding…")
    embedding_model = Inference(
        Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token),
        window="whole",
    )
    log.info("✅ pyannote embedding chargé")

    log.info("Chargement Silero VAD…")
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = vad_utils
    log.info("✅ Silero VAD chargé")

    log.info("Chargement faster-whisper (%s, %s)…", whisper_model_size, device)
    from faster_whisper import WhisperModel

    compute_type = "float16" if device == "cuda" else "int8"
    whisper = WhisperModel(whisper_model_size, device=device, compute_type=compute_type)
    log.info("✅ faster-whisper %s chargé (%s, %s)", whisper_model_size, device, compute_type)

    return {
        "diarization": diarization_pipeline,
        "embedding": embedding_model,
        "vad": vad_model,
        "vad_get_timestamps": get_speech_timestamps,
        "whisper": whisper,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def convert_audio(filepath: str, out_path: str) -> float:
    """Convertit n'importe quel format audio en WAV mono 16 kHz."""
    audio = AudioSegment.from_file(filepath)
    audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
    audio.export(out_path, format="wav")
    duration = len(audio) / 1000
    log.info("✅ Converti → %s (%.1fs)", out_path, duration)
    return duration


def run_vad(audio_1d: torch.Tensor, vad_model, get_speech_timestamps) -> list:
    return get_speech_timestamps(
        audio_1d,
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=0.4,
        min_speech_duration_ms=200,
        min_silence_duration_ms=100,
    )


def run_diarization(audio_path: str, pipeline, num_speakers=None) -> list:
    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers

    diarization = pipeline(audio_path, **kwargs)
    segments = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "speaker": speaker,
            }
        )
    return segments


def transcribe(chunk_path: str, whisper_model, language: str = "fr") -> str:
    segs, _ = whisper_model.transcribe(chunk_path, language=language, beam_size=5)
    return " ".join([s.text for s in segs]).strip().lower()


def get_speaker_at(t: float, segments: list, default: str = "UNKNOWN") -> str:
    for seg in segments:
        if seg["start"] <= t <= seg["end"]:
            return seg["speaker"]
    return default


def get_embedding(wav_path: str, embedding_model):
    return embedding_model({"uri": "chunk", "audio": wav_path})


def cosine_similarity(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ─────────────────────────────────────────────────────────────────────────────
# Étapes de la pipeline
# ─────────────────────────────────────────────────────────────────────────────

def step_vad(audio_1d: torch.Tensor, audio_array: np.ndarray, models: dict, workdir: str):
    log.info("── VAD ──────────────────────────────────────────────")
    speech_timestamps = run_vad(audio_1d, models["vad"], models["vad_get_timestamps"])
    log.info("🔊 VAD : %d segment(s) détecté(s)", len(speech_timestamps))
    for i, ts in enumerate(speech_timestamps):
        s = ts["start"] / SAMPLE_RATE
        e = ts["end"] / SAMPLE_RATE
        log.info("  Seg %02d : [%.2fs → %.2fs] (%.2fs)", i + 1, s, e, e - s)
    return speech_timestamps


def step_diarization(audio_path: str, models: dict, num_speakers=None):
    log.info("── Diarisation ──────────────────────────────────────")
    segments = run_diarization(audio_path, models["diarization"], num_speakers)
    speakers = list(set(s["speaker"] for s in segments))
    log.info("🎙️  %d locuteur(s) : %s", len(speakers), speakers)
    return segments, speakers


def step_wake_word(
    speech_timestamps: list,
    audio_array: np.ndarray,
    diarization_segments: list,
    models: dict,
    workdir: str,
    language: str,
):
    log.info("── Wake word ─────────────────────────────────────────")
    triggers = []
    for i, ts in enumerate(speech_timestamps):
        start_s = ts["start"] / SAMPLE_RATE
        end_s = ts["end"] / SAMPLE_RATE
        chunk = audio_array[ts["start"] : ts["end"]]
        chunk_path = os.path.join(workdir, f"chunk_{i:02d}.wav")
        sf.write(chunk_path, chunk, SAMPLE_RATE)

        text = transcribe(chunk_path, models["whisper"], language)
        is_trigger = any(v in text for v in WAKE_WORD_VARIANTS)
        speaker = get_speaker_at(
            start_s + (end_s - start_s) / 2, diarization_segments
        )

        status = "🔔 OUI" if is_trigger else "—"
        log.info('  Seg %02d [%.1fs] "%s"  %s  %s', i + 1, end_s - start_s, text[:42], status, speaker)

        if is_trigger:
            triggers.append(
                {
                    "segment_id": i + 1,
                    "timestamp_start": round(start_s, 3),
                    "timestamp_end": round(end_s, 3),
                    "trigger_word": next(v for v in WAKE_WORD_VARIANTS if v in text),
                    "transcript": text,
                    "active_speaker": speaker,
                }
            )

    log.info("✅ %d déclenchement(s)", len(triggers))
    return triggers


def step_enrollment(
    speech_timestamps: list,
    audio_array: np.ndarray,
    diarization_segments: list,
    speakers_found: list,
    models: dict,
    workdir: str,
    language: str,
    enroll_path: str = None,
):
    """
    Enrôlement conducteur/passager.

    - Si enroll_path est fourni : utilise les 2 premiers segments VAD de ce fichier.
    - Sinon : utilise les 2 premiers segments VAD de l'audio principal.
    """
    log.info("── Enrôlement conducteur/passager ────────────────────")

    if len(speech_timestamps) < 2:
        log.warning("⚠️  Moins de 2 segments VAD — enrôlement ignoré")
        return {}, {}

    roles = ["conducteur", "passager"]
    enrollment_embeddings = {}

    if enroll_path:
        log.info("Enrôlement depuis fichier dédié : %s", enroll_path)
        enroll_wav = os.path.join(workdir, "enroll_converted.wav")
        convert_audio(enroll_path, enroll_wav)
        waveform_e, _ = torchaudio.load(enroll_wav)
        audio_e = waveform_e.squeeze(0).numpy()

        # VAD sur le fichier d'enrôlement
        audio_e_tensor = torch.tensor(audio_e)
        ts_e = run_vad(audio_e_tensor, models["vad"], models["vad_get_timestamps"])
        source_timestamps = ts_e
        source_array = audio_e
    else:
        source_timestamps = speech_timestamps
        source_array = audio_array

    for idx in range(min(2, len(source_timestamps))):
        ts = source_timestamps[idx]
        chunk = source_array[ts["start"] : ts["end"]]
        cp = os.path.join(workdir, f"enrol_{roles[idx]}.wav")
        sf.write(cp, chunk, SAMPLE_RATE)
        text = transcribe(cp, models["whisper"], language)
        log.info('  %s — segment %d : "%s"', roles[idx].upper(), idx + 1, text)
        enrollment_embeddings[roles[idx]] = get_embedding(cp, models["embedding"])

    # Embeddings moyens par locuteur
    speaker_embeddings = {}
    for spk in speakers_found:
        chunks_emb = []
        for seg in diarization_segments:
            if seg["speaker"] == spk:
                chunk = audio_array[
                    int(seg["start"] * SAMPLE_RATE) : int(seg["end"] * SAMPLE_RATE)
                ]
                if len(chunk) > SAMPLE_RATE * 0.5:
                    cp = os.path.join(workdir, f"spk_{spk}_{len(chunks_emb)}.wav")
                    sf.write(cp, chunk, SAMPLE_RATE)
                    chunks_emb.append(get_embedding(cp, models["embedding"]))
        if chunks_emb:
            speaker_embeddings[spk] = np.mean(chunks_emb, axis=0)

    # Décision : 1 conducteur, 1 passager, reste = occupant
    role_mapping = {}
    best_conducteur = max(
        speaker_embeddings,
        key=lambda spk: cosine_similarity(speaker_embeddings[spk], enrollment_embeddings["conducteur"]),
    )

    remaining = [s for s in speaker_embeddings if s != best_conducteur]
    best_passager = (
        max(
            remaining,
            key=lambda spk: cosine_similarity(speaker_embeddings[spk], enrollment_embeddings["passager"]),
        )
        if remaining
        else None
    )

    log.info("📊 Similarité cosinus :")
    for spk, emb in speaker_embeddings.items():
        sim_c = cosine_similarity(emb, enrollment_embeddings["conducteur"])
        sim_p = cosine_similarity(emb, enrollment_embeddings["passager"])
        if spk == best_conducteur:
            role = "conducteur"
        elif spk == best_passager:
            role = "passager"
        else:
            role = "occupant"
        role_mapping[spk] = role
        log.info("  %s  sim_c=%.4f  sim_p=%.4f  → %s", spk, sim_c, sim_p, role.upper())

    return role_mapping, enrollment_embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principale
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    audio_file: str,
    hf_token: str,
    output_path: str = None,
    enroll_path: str = None,
    num_speakers: int = None,
    language: str = "fr",
    whisper_model_size: str = "base",
    device: str = "cpu",
    workdir: str = None,
):
    own_workdir = workdir is None
    if own_workdir:
        workdir = tempfile.mkdtemp(prefix="pa10_")
    os.makedirs(workdir, exist_ok=True)

    # ── Conversion audio ─────────────────────────────────────────────────────
    log.info("── Conversion audio ─────────────────────────────────")
    wav_path = os.path.join(workdir, "pipeline_audio.wav")
    duration = convert_audio(audio_file, wav_path)

    # ── Chargement modèles ───────────────────────────────────────────────────
    models = load_models(hf_token, whisper_model_size=whisper_model_size, device=device)

    # ── Lecture waveform ─────────────────────────────────────────────────────
    waveform, sr = torchaudio.load(wav_path)
    audio_1d = waveform.squeeze(0)
    audio_array = audio_1d.numpy()

    # ── Étapes ───────────────────────────────────────────────────────────────
    speech_timestamps = step_vad(audio_1d, audio_array, models, workdir)
    diarization_segments, speakers_found = step_diarization(wav_path, models, num_speakers)
    triggers = step_wake_word(speech_timestamps, audio_array, diarization_segments, models, workdir, language)
    role_mapping, _ = step_enrollment(
        speech_timestamps, audio_array, diarization_segments, speakers_found,
        models, workdir, language, enroll_path
    )

    # ── Export JSON ──────────────────────────────────────────────────────────
    output = {
        "metadata": {
            "story": "PA-10",
            "pipeline": "complète",
            "audio_file": os.path.basename(audio_file),
            "duration_s": round(duration, 3),
            "processed_at": datetime.datetime.now().isoformat(),
            "vad_model": "silero-vad",
            "diarization_model": "pyannote/speaker-diarization-3.1",
            "embedding_model": "pyannote/embedding",
            "transcription_model": f"faster-whisper-{whisper_model_size}-int8",
            "wake_word_variants": WAKE_WORD_VARIANTS,
            "speakers_detected": speakers_found,
            "num_speakers_detected": len(speakers_found),
            "num_vad_segments": len(speech_timestamps),
            "modules_integres": ["VAD", "diarisation", "transcription", "wake_word", "conducteur_passager"],
            "modules_a_venir": ["detection_emotion", "detection_genre", "DER"],
        },
        "diarization_segments": diarization_segments,
        "wake_word_triggers": triggers,
        "role_mapping": role_mapping,
        "der_score": None,
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        log.info("📄 JSON exporté → %s", output_path)
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))

    # ── Récapitulatif ─────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  RÉCAPITULATIF PA-10")
    log.info("=" * 60)
    log.info("  Audio         : %s (%.1fs)", os.path.basename(audio_file), duration)
    log.info("  Locuteurs     : %d → %s", len(speakers_found), speakers_found)
    log.info("  Segments VAD  : %d", len(speech_timestamps))
    log.info("  Segments diar : %d", len(diarization_segments))
    log.info("  Wake words    : %d", len(triggers))
    for spk, role in role_mapping.items():
        log.info("  %s → %s", spk, role.upper())
    log.info("=" * 60)

    return output


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PA-10 — Pipeline audio complète (VAD + diarisation + wake word + enrôlement)"
    )
    parser.add_argument("audio", help="Fichier audio à traiter (wav, mp3, flac, ogg…)")
    parser.add_argument("--output", "-o", help="Chemin du JSON de sortie (défaut : stdout)")
    parser.add_argument("--enroll", help="Fichier audio d'enrôlement conducteur/passager séparé")
    parser.add_argument("--num-speakers", type=int, default=None, help="Nombre de locuteurs attendu")
    parser.add_argument("--language", default="fr", help="Langue pour Whisper (défaut : fr)")
    parser.add_argument("--whisper-model", default="base", help="Taille du modèle Whisper (tiny/base/small)")
    parser.add_argument("--workdir", default=None, help="Répertoire de travail temporaire")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        log.error("❌ HF_TOKEN non défini. Exportez-le ou utilisez un fichier .env")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device : %s", device)

    run_pipeline(
        audio_file=args.audio,
        hf_token=hf_token,
        output_path=args.output,
        enroll_path=args.enroll,
        num_speakers=args.num_speakers,
        language=args.language,
        whisper_model_size=args.whisper_model,
        device=device,
        workdir=args.workdir,
    )


if __name__ == "__main__":
    main()
