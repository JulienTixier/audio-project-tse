#!/usr/bin/env python3
"""
PA-10 - Pipeline Complète de Traitement Audio
Sprint 5 | Équipe P07 | Projet Stellantis

Modules : VAD (Silero) + Diarisation (pyannote) + Transcription (faster-whisper)
          + Wake word + Enrolement conducteur/passager
          + Genre/Âge (pitch F0 + audeering wav2vec2)
          + DER (optionnel, via fichier RTTM)
          + Export JSON

Usage:
    python process.py <audio_file> [--output <out.json>] [--enroll <enroll.wav>]
                      [--reference <verite.rttm>] [--num-speakers <n>]
                      [--language <lang>] [--whisper-model tiny|base|small]
"""

import argparse
import datetime
import json
import logging
import os
import sys
import tempfile
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from pydub import AudioSegment
from scipy.signal import find_peaks

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

AG_MODEL_NAME = "audeering/wav2vec2-large-robust-6-ft-age-gender"


# -----------------------------------------------------------------------------
# Architecture modèle Genre/Âge (audeering)
# -----------------------------------------------------------------------------

class _ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense    = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout  = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class _AgeGenderModel(nn.Module):
    def __init__(self, config):
        from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age      = _ModelHead(config, 1)
        self.gender   = _ModelHead(config, 3)

    def forward(self, input_values):
        hidden = self.wav2vec2(input_values)[0]
        hidden = torch.mean(hidden, dim=1)
        return hidden, self.age(hidden), self.gender(hidden)


# -----------------------------------------------------------------------------
# Chargement des modèles
# -----------------------------------------------------------------------------

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

    log.info("Chargement pyannote diarisation...")
    from pyannote.audio import Pipeline, Model, Inference

    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )
    diarization_pipeline = diarization_pipeline.to(torch.device(device))
    log.info("[OK] pyannote diarisation charge")

    log.info("Chargement pyannote embedding...")
    embedding_model = Inference(
        Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token),
        window="whole",
    )
    log.info("[OK] pyannote embedding charge")

    log.info("Chargement Silero VAD...")
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = vad_utils
    log.info("[OK] Silero VAD charge")

    log.info("Chargement faster-whisper (%s, %s)...", whisper_model_size, device)
    from faster_whisper import WhisperModel
    compute_type = "float16" if device == "cuda" else "int8"
    whisper = WhisperModel(whisper_model_size, device=device, compute_type=compute_type)
    log.info("[OK] faster-whisper %s charge (%s, %s)", whisper_model_size, device, compute_type)

    log.info("Chargement audeering genre+age...")
    from transformers import Wav2Vec2Processor, Wav2Vec2Config
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download

    ag_processor = Wav2Vec2Processor.from_pretrained(AG_MODEL_NAME)
    ag_config    = Wav2Vec2Config.from_pretrained(AG_MODEL_NAME)
    ag_model     = _AgeGenderModel(ag_config)
    ag_weights   = hf_hub_download(repo_id=AG_MODEL_NAME, filename="model.safetensors")
    ag_model.load_state_dict(load_file(ag_weights), strict=False)
    ag_model     = ag_model.to(torch.device(device))
    ag_model.eval()
    log.info("[OK] audeering genre+age charge")

    return {
        "diarization":        diarization_pipeline,
        "embedding":          embedding_model,
        "vad":                vad_model,
        "vad_get_timestamps": get_speech_timestamps,
        "whisper":            whisper,
        "ag_processor":       ag_processor,
        "ag_model":           ag_model,
        "ag_device":          torch.device(device),
    }


# -----------------------------------------------------------------------------
# Fonctions utilitaires
# -----------------------------------------------------------------------------

def convert_audio(filepath: str, out_path: str) -> float:
    """Convertit n'importe quel format audio en WAV mono 16 kHz."""
    audio = AudioSegment.from_file(filepath)
    audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
    audio.export(out_path, format="wav")
    duration = len(audio) / 1000
    log.info("[OK] Converti en %s (%.1fs)", out_path, duration)
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
    log.info(" Calcul de la diarisation en cours (peut prendre plusieurs minutes sur CPU)...")
    diarization = pipeline(audio_path, **kwargs)
    segments = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start":   round(segment.start, 3),
            "end":     round(segment.end,   3),
            "speaker": speaker,
        })
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


def estimate_f0(audio_array: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Estimation du pitch F0 par autocorrélation."""
    frame_size = int(0.025 * sample_rate)
    hop_size   = int(0.010 * sample_rate)
    min_period = int(sample_rate / 400)
    max_period = int(sample_rate / 50)
    f0_values  = []
    for start in range(0, len(audio_array) - frame_size, hop_size):
        frame = audio_array[start : start + frame_size]
        frame = frame - np.mean(frame)
        if np.max(np.abs(frame)) < 0.01:
            continue
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2:]
        peaks, _ = find_peaks(corr[min_period:max_period])
        if len(peaks) > 0:
            best_period = peaks[np.argmax(corr[min_period:max_period][peaks])] + min_period
            f0 = sample_rate / best_period
            if 50 < f0 < 400:
                f0_values.append(f0)
    return np.array(f0_values)


def predict_gender_age(audio_array: np.ndarray, models: dict) -> dict:
    """
    Genre via pitch F0 (robuste en français), âge via audeering wav2vec2.
    Retourne un dict : gender, confidence, pitch_f0_mean, age_estimate.
    """
    # -- Genre (F0) ------------------------------------------------------------
    f0_values = estimate_f0(audio_array)
    if len(f0_values) > 0:
        mean_f0    = float(np.mean(f0_values))
        gender     = "male" if mean_f0 < 190 else "female"
        confidence = min(abs(mean_f0 - 190) / 95, 1.0)
    else:
        mean_f0, gender, confidence = 0.0, "unknown", 0.0

    # -- Âge (audeering) -------------------------------------------------------
    processor = models["ag_processor"]
    ag_model  = models["ag_model"]
    ag_device = models["ag_device"]

    y = processor(audio_array, sampling_rate=SAMPLE_RATE)["input_values"][0]
    y = torch.from_numpy(y.reshape(1, -1)).to(ag_device)
    with torch.no_grad():
        _, logits_age, _ = ag_model(y)
    age = float(logits_age[0][0]) * 100

    return {
        "gender":        gender,
        "confidence":    round(confidence, 4),
        "pitch_f0_mean": round(mean_f0, 1),
        "age_estimate":  round(age, 1),
    }


def compute_der(reference_rttm_path: str, diarization_segments: list) -> float:
    """Calcule le DER entre un fichier RTTM de référence et la diarisation."""
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    reference = Annotation()
    with open(reference_rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start    = float(parts[3])
            duration = float(parts[4])
            speaker  = parts[7]
            reference[Segment(start, start + duration)] = speaker

    hypothesis = Annotation()
    for seg in diarization_segments:
        hypothesis[Segment(seg["start"], seg["end"])] = seg["speaker"]

    metric = DiarizationErrorRate()
    der    = metric(reference, hypothesis)
    return round(der * 100, 2)


# -----------------------------------------------------------------------------
# Étapes de la pipeline
# -----------------------------------------------------------------------------

def step_vad(audio_1d: torch.Tensor, models: dict):
    log.info("-- VAD ----------------------------------------------")
    speech_timestamps = run_vad(audio_1d, models["vad"], models["vad_get_timestamps"])
    log.info("VAD : %d segment(s) detecte(s)", len(speech_timestamps))
    for i, ts in enumerate(speech_timestamps):
        s = ts["start"] / SAMPLE_RATE
        e = ts["end"]   / SAMPLE_RATE
        log.info("  Seg %02d : [%.2fs -> %.2fs] (%.2fs)", i + 1, s, e, e - s)
    return speech_timestamps


def step_diarization(audio_path: str, models: dict, num_speakers=None):
    log.info("-- Diarisation --------------------------------------")
    segments = run_diarization(audio_path, models["diarization"], num_speakers)
    speakers = list(set(s["speaker"] for s in segments))
    log.info("%d locuteur(s) : %s", len(speakers), speakers)
    return segments, speakers


def step_wake_word(
    speech_timestamps: list,
    audio_array: np.ndarray,
    diarization_segments: list,
    models: dict,
    workdir: str,
    language: str,
):
    log.info("-- Wake word -----------------------------------------")
    triggers = []
    for i, ts in enumerate(speech_timestamps):
        start_s = ts["start"] / SAMPLE_RATE
        end_s   = ts["end"]   / SAMPLE_RATE
        chunk   = audio_array[ts["start"] : ts["end"]]
        chunk_path = os.path.join(workdir, f"chunk_{i:02d}.wav")
        sf.write(chunk_path, chunk, SAMPLE_RATE)

        text       = transcribe(chunk_path, models["whisper"], language)
        is_trigger = any(v in text for v in WAKE_WORD_VARIANTS)
        speaker    = get_speaker_at(start_s + (end_s - start_s) / 2, diarization_segments)
        status     = "[TRIGGER]" if is_trigger else "-"
        log.info('  Seg %02d [%.1fs] "%s"  %s  %s', i + 1, end_s - start_s, text[:42], status, speaker)

        if is_trigger:
            triggers.append({
                "segment_id":      i + 1,
                "timestamp_start": round(start_s, 3),
                "timestamp_end":   round(end_s,   3),
                "trigger_word":    next(v for v in WAKE_WORD_VARIANTS if v in text),
                "transcript":      text,
                "active_speaker":  speaker,
            })

    log.info("[OK] %d declenchement(s)", len(triggers))
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
    Enrolement conducteur/passager.
    - Sans --enroll : utilise les 2 premiers segments VAD de l'audio principal.
    - Avec --enroll : utilise les 2 premiers segments VAD du fichier dédié.
    """
    log.info("-- Enrolement conducteur/passager --------------------")

    if len(speech_timestamps) < 2:
        log.warning("[WARN] Moins de 2 segments VAD - enrolement ignore")
        return {}, {}

    roles = ["conducteur", "passager"]
    enrollment_embeddings = {}

    if enroll_path:
        log.info("Enrolement depuis fichier dedie : %s", enroll_path)
        enroll_wav = os.path.join(workdir, "enroll_converted.wav")
        convert_audio(enroll_path, enroll_wav)
        waveform_e, _ = torchaudio.load(enroll_wav)
        audio_e = waveform_e.squeeze(0).numpy()
        ts_e = run_vad(torch.tensor(audio_e), models["vad"], models["vad_get_timestamps"])
        source_timestamps = ts_e
        source_array      = audio_e
    else:
        source_timestamps = speech_timestamps
        source_array      = audio_array

    for idx in range(min(2, len(source_timestamps))):
        ts    = source_timestamps[idx]
        chunk = source_array[ts["start"] : ts["end"]]
        cp    = os.path.join(workdir, f"enrol_{roles[idx]}.wav")
        sf.write(cp, chunk, SAMPLE_RATE)
        text  = transcribe(cp, models["whisper"], language)
        log.info('  %s - segment %d : "%s"', roles[idx].upper(), idx + 1, text)
        enrollment_embeddings[roles[idx]] = get_embedding(cp, models["embedding"])

    # Embeddings moyens par locuteur
    speaker_embeddings = {}
    for spk in speakers_found:
        chunks_emb = []
        for seg in diarization_segments:
            if seg["speaker"] == spk:
                chunk = audio_array[int(seg["start"] * SAMPLE_RATE) : int(seg["end"] * SAMPLE_RATE)]
                if len(chunk) > SAMPLE_RATE * 0.5:
                    cp = os.path.join(workdir, f"spk_{spk}_{len(chunks_emb)}.wav")
                    sf.write(cp, chunk, SAMPLE_RATE)
                    chunks_emb.append(get_embedding(cp, models["embedding"]))
        if chunks_emb:
            speaker_embeddings[spk] = np.mean(chunks_emb, axis=0)

    # Décision : 1 conducteur, 1 passager, reste = occupant
    role_mapping    = {}
    best_conducteur = max(
        speaker_embeddings,
        key=lambda spk: cosine_similarity(speaker_embeddings[spk], enrollment_embeddings["conducteur"]),
    )
    remaining   = [s for s in speaker_embeddings if s != best_conducteur]
    best_passager = (
        max(remaining, key=lambda spk: cosine_similarity(speaker_embeddings[spk], enrollment_embeddings["passager"]))
        if remaining else None
    )

    log.info("Similarite cosinus :")
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
        log.info("  %s  sim_c=%.4f  sim_p=%.4f  -> %s", spk, sim_c, sim_p, role.upper())

    return role_mapping, enrollment_embeddings


def step_gender_age(
    audio_array: np.ndarray,
    diarization_segments: list,
    speakers_found: list,
    role_mapping: dict,
    models: dict,
    workdir: str,
):
    """Analyse genre (F0) + âge (audeering) par locuteur."""
    log.info("-- Genre + Age ---------------------------------------")
    gender_results = {}

    for spk in speakers_found:
        chunks         = []
        total_duration = 0.0
        for seg in diarization_segments:
            if seg["speaker"] == spk:
                start = int(seg["start"] * SAMPLE_RATE)
                end   = int(seg["end"]   * SAMPLE_RATE)
                chunk = audio_array[start:end]
                if len(chunk) > SAMPLE_RATE * 0.3:
                    chunks.append(chunk)
                    total_duration += len(chunk) / SAMPLE_RATE

        if not chunks:
            log.warning("  [WARN] %s - pas assez de signal audio", spk)
            continue

        full_audio = np.concatenate(chunks)
        spk_path   = os.path.join(workdir, f"spk_gender_{spk}.wav")
        sf.write(spk_path, full_audio, SAMPLE_RATE)

        result = predict_gender_age(full_audio, models)
        role   = role_mapping.get(spk, "occupant")
        icon   = "[F]" if result["gender"] == "female" else "[M]"

        gender_results[spk] = {
            **result,
            "role":           role,
            "total_speech_s": round(total_duration, 2),
        }

        log.info(
            "  %s %s [%s] -> %s (F0: %.0fHz | confiance: %.3f) | âge: %.0f ans | parole: %.1fs",
            icon, spk, role.upper(), result["gender"].upper(),
            result["pitch_f0_mean"], result["confidence"],
            result["age_estimate"], total_duration,
        )

    log.info("[OK] Analyse genre+age terminee")
    return gender_results


# -----------------------------------------------------------------------------
# Pipeline principale
# -----------------------------------------------------------------------------

def run_pipeline(
    audio_file: str,
    hf_token: str,
    output_path: str    = None,
    enroll_path: str    = None,
    reference_rttm: str = None,
    num_speakers: int   = None,
    language: str       = "fr",
    whisper_model_size: str = "base",
    device: str         = "cpu",
    workdir: str        = None,
):
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix="pa10_")
    os.makedirs(workdir, exist_ok=True)

    t_pipeline_start = time.perf_counter()
    latency = {}

    # -- Conversion audio -----------------------------------------------------
    log.info("-- Conversion audio ---------------------------------")
    wav_path = os.path.join(workdir, "pipeline_audio.wav")
    _t0 = time.perf_counter()
    duration = convert_audio(audio_file, wav_path)
    latency["conversion_audio_s"] = round(time.perf_counter() - _t0, 3)

    # -- Chargement modèles ---------------------------------------------------
    _t0 = time.perf_counter()
    models = load_models(hf_token, whisper_model_size=whisper_model_size, device=device)
    latency["chargement_modeles_s"] = round(time.perf_counter() - _t0, 3)

    # -- Lecture waveform -----------------------------------------------------
    waveform, _ = torchaudio.load(wav_path)
    audio_1d    = waveform.squeeze(0)
    audio_array = audio_1d.numpy()

    # -- Étapes ---------------------------------------------------------------
    _t0 = time.perf_counter()
    speech_timestamps = step_vad(audio_1d, models)
    latency["vad_s"] = round(time.perf_counter() - _t0, 3)

    _t0 = time.perf_counter()
    diarization_segments, speakers_found = step_diarization(wav_path, models, num_speakers)
    latency["diarisation_s"] = round(time.perf_counter() - _t0, 3)

    _t0 = time.perf_counter()
    triggers = step_wake_word(speech_timestamps, audio_array, diarization_segments, models, workdir, language)
    latency["wake_word_s"] = round(time.perf_counter() - _t0, 3)

    _t0 = time.perf_counter()
    role_mapping, _ = step_enrollment(speech_timestamps, audio_array, diarization_segments, speakers_found, models, workdir, language, enroll_path)
    latency["enrolement_s"] = round(time.perf_counter() - _t0, 3)

    _t0 = time.perf_counter()
    gender_results = step_gender_age(audio_array, diarization_segments, speakers_found, role_mapping, models, workdir)
    latency["genre_age_s"] = round(time.perf_counter() - _t0, 3)

    # -- DER (optionnel) ------------------------------------------------------
    der_score = None
    if reference_rttm:
        log.info("-- DER ----------------------------------------------")
        _t0 = time.perf_counter()
        der_score = compute_der(reference_rttm, diarization_segments)
        latency["der_s"] = round(time.perf_counter() - _t0, 3)
        log.info("DER : %.2f%%", der_score)
        if der_score <= 30:
            log.info("[OK] DER acceptable pour un prototype mono-micro open source")
        else:
            log.warning("[WARN] DER eleve (%.2f%%) - pipeline a optimiser", der_score)

    latency["total_s"] = round(time.perf_counter() - t_pipeline_start, 3)

    # -- Export JSON ----------------------------------------------------------
    output = {
        "metadata": {
            "story":               "PA-10",
            "pipeline":            "complète",
            "audio_file":          os.path.basename(audio_file),
            "duration_s":          round(duration, 3),
            "processed_at":        datetime.datetime.now().isoformat(),
            "vad_model":           "silero-vad",
            "diarization_model":   "pyannote/speaker-diarization-3.1",
            "embedding_model":     "pyannote/embedding",
            "transcription_model": f"faster-whisper-{whisper_model_size}-int8",
            "genre_model":         "pitch-F0-autocorrelation + audeering-wav2vec2",
            "detection_emotion":   "non implémenté",
            "wake_word_variants":  WAKE_WORD_VARIANTS,
            "speakers_detected":   speakers_found,
            "num_speakers_detected": len(speakers_found),
            "num_vad_segments":    len(speech_timestamps),
            "modules_integres":    ["VAD", "diarisation", "transcription", "wake_word", "conducteur_passager", "genre_age", "DER"],
            "modules_a_venir":     ["detection_emotion"],
        },
        "diarization_segments":    diarization_segments,
        "wake_word_triggers":      triggers,
        "role_mapping":            role_mapping,
        "genre_age_par_locuteur":  gender_results,
        "der_score":               der_score,
        "latency": latency,
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        log.info("JSON exporte vers %s", output_path)
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))

    # -- Récapitulatif ---------------------------------------------------------
    log.info("=" * 60)
    log.info("  RECAPITULATIF PA-10")
    log.info("=" * 60)
    log.info("  Audio         : %s (%.1fs)", os.path.basename(audio_file), duration)
    log.info("  Locuteurs     : %d -> %s", len(speakers_found), speakers_found)
    log.info("  Segments VAD  : %d", len(speech_timestamps))
    log.info("  Segments diar : %d", len(diarization_segments))
    log.info("  Wake words    : %d", len(triggers))
    log.info("  DER           : %s", f"{der_score:.2f}%" if der_score is not None else "non calcule (pas de --reference)")
    log.info("  Modules : VAD OK | Diarisation OK | Transcription OK | Wake word OK | Conducteur/Passager OK | Genre+Age OK | DER %s", "OK" if der_score is not None else "(optionnel)")
    log.info("-- Latences -----------------------------------------")
    for key, val in latency.items():
        if key != "total_s":
            log.info("  %-30s %.3f s", key, val)
    log.info("  %-30s %.3f s  (total pipeline)", "total_s", latency["total_s"])
    for spk in speakers_found:
        role = role_mapping.get(spk, "?")
        gr   = gender_results.get(spk, {})
        icon = "[F]" if gr.get("gender") == "female" else "[M]"
        log.info("  %s %s -> %s | %s (F0: %.0fHz) | %.0f ans",
                 icon, spk, role.upper(),
                 gr.get("gender", "?").upper(), gr.get("pitch_f0_mean", 0), gr.get("age_estimate", 0))
    log.info("=" * 60)

    return output


# -----------------------------------------------------------------------------
# Point d'entrée
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PA-10 - Pipeline audio complète (VAD + diarisation + wake word + enrôlement + genre/âge + DER)"
    )
    parser.add_argument("audio",              help="Fichier audio à traiter (wav, mp3, flac, ogg...)")
    parser.add_argument("--output",    "-o",  help="Chemin du JSON de sortie (défaut : stdout)")
    parser.add_argument("--enroll",           help="Fichier audio d'enrôlement conducteur/passager séparé")
    parser.add_argument("--reference",        help="Fichier RTTM de vérité terrain pour le calcul du DER (optionnel)")
    parser.add_argument("--num-speakers",     type=int, default=None, help="Nombre de locuteurs attendu")
    parser.add_argument("--language",         default="fr", help="Langue pour Whisper (défaut : fr)")
    parser.add_argument("--whisper-model",    default="base", help="Taille du modèle Whisper : tiny / base / small")
    parser.add_argument("--workdir",          default=None, help="Répertoire de travail temporaire")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        log.error("[ERREUR] HF_TOKEN non defini.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device : %s", device)

    run_pipeline(
        audio_file      = args.audio,
        hf_token        = hf_token,
        output_path     = args.output,
        enroll_path     = args.enroll,
        reference_rttm  = args.reference,
        num_speakers    = args.num_speakers,
        language        = args.language,
        whisper_model_size = args.whisper_model,
        device          = device,
        workdir         = args.workdir,
    )


if __name__ == "__main__":
    main()
