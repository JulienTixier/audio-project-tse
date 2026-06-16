#!/usr/bin/env python3
"""
Audio diarization pipeline — VAD (Silero) + Speaker diarization (pyannote 3.1)
Usage: python process.py <audio_file> [--output <json_file>]
"""

import argparse
import datetime
import json
import os
import sys

import librosa
import soundfile as sf
import torch
import torchaudio

# ── Patch torchaudio si nécessaire (version mismatch) ─────────────────────────
if not hasattr(torchaudio, "AudioMetaData"):
    try:
        from torchaudio._backend.utils import AudioMetaData
    except Exception:
        import collections
        AudioMetaData = collections.namedtuple(
            "AudioMetaData",
            ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
        )
    torchaudio.AudioMetaData = AudioMetaData

from pyannote.audio import Pipeline

SAMPLE_RATE = 16000


def load_and_convert(audio_path: str) -> tuple[torch.Tensor, str]:
    """Charge n'importe quel format audio, convertit en wav 16 kHz mono."""
    tmp_wav = "/tmp/converted_audio.wav"
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    sf.write(tmp_wav, audio, SAMPLE_RATE)
    waveform, sr = torchaudio.load(tmp_wav)
    assert sr == SAMPLE_RATE
    print(f"Audio chargé : {waveform.shape[1] / sr:.1f}s | {sr} Hz | mono", flush=True)
    return waveform, tmp_wav


def run_vad(waveform: torch.Tensor) -> list[dict]:
    """Détection d'activité vocale avec Silero VAD."""
    print("Chargement Silero VAD...", flush=True)

    # Silero est téléchargé depuis torch.hub — le cache est dans TORCH_HOME
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    (get_speech_timestamps, _, _, _, _) = utils

    audio_1d = waveform.squeeze(0)
    speech_timestamps = get_speech_timestamps(
        audio_1d,
        model,
        sampling_rate=SAMPLE_RATE,
        threshold=0.5,
        min_speech_duration_ms=300,
        min_silence_duration_ms=200,
    )
    print(f"VAD : {len(speech_timestamps)} segment(s) détecté(s)", flush=True)
    return speech_timestamps


def run_diarization(wav_path: str, hf_token: str) -> list[dict]:
    """Diarisation avec pyannote.audio 3.1."""
    print("Chargement pyannote pipeline...", flush=True)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    # CPU uniquement sur Raspberry Pi — à changer en .to("cuda") sur Jetson Orin
    pipeline = pipeline.to(torch.device("cpu"))

    print("Inférence pyannote (peut prendre quelques minutes sur CPU)...", flush=True)
    diarization = pipeline(wav_path)

    segments = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "speaker": speaker,
            }
        )
    print(f"Diarisation : {len(segments)} segment(s)", flush=True)
    return segments


def build_output(audio_path: str, waveform: torch.Tensor, vad: list, segments: list) -> dict:
    duration = round(waveform.shape[1] / SAMPLE_RATE, 3)
    speakers = set(s["speaker"] for s in segments)
    return {
        "metadata": {
            "audio_file": os.path.basename(audio_path),
            "duration_s": duration,
            "processed_at": datetime.datetime.now().isoformat(),
            "num_speakers_detected": len(speakers),
            "num_vad_segments": len(vad),
        },
        "segments": segments,
    }


def main():
    parser = argparse.ArgumentParser(description="VAD + Speaker diarization pipeline")
    parser.add_argument("audio", help="Chemin vers le fichier audio (wav, flac, mp3, ogg…)")
    parser.add_argument("--output", "-o", default=None, help="Fichier JSON de sortie (optionnel)")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Erreur : la variable d'environnement HF_TOKEN est manquante.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.audio):
        print(f"Erreur : fichier introuvable : {args.audio}", file=sys.stderr)
        sys.exit(1)

    print(f"\nTraitement de : {args.audio}", flush=True)

    waveform, tmp_wav = load_and_convert(args.audio)
    vad_segments = run_vad(waveform)
    diar_segments = run_diarization(tmp_wav, hf_token)
    output = build_output(args.audio, waveform, vad_segments, diar_segments)

    json_str = json.dumps(output, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"\nRésultat écrit dans : {args.output}", flush=True)
    else:
        print("\n── Résultat JSON ──────────────────────────────────────────")
        print(json_str)

    # Validation PA-28
    c1 = len(vad_segments) > 0
    c2 = len(diar_segments) > 0
    print("\n── Validation PA-28 ───────────────────────────────────────")
    print(f"  Critère 1 — Pipeline sans plantage     : {'PASS ✔' if c1 and c2 else 'FAIL ✘'}")
    print(f"  Critère 2 — Sortie brute voix+segments : {'PASS ✔' if c2 else 'FAIL ✘'}")
    print(f"  Locuteurs détectés : {output['metadata']['num_speakers_detected']}")
    print(f"  Segments produits  : {len(diar_segments)}")


if __name__ == "__main__":
    main()
