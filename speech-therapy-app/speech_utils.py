import functools
import re

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from constants import FILLER_WORDS, IDEAL_WPM_RANGE, LANGUAGE_MODELS, TARGET_SR, COMMON_WORD_WHITELIST


@functools.lru_cache(maxsize=4)
def _load_wav2vec2(checkpoint: str):
    processor = Wav2Vec2Processor.from_pretrained(checkpoint)
    model = Wav2Vec2ForCTC.from_pretrained(checkpoint)
    model.eval()
    return processor, model


def transcribe_audio(audio_path: str, language: str = "English") -> dict:
    checkpoint = LANGUAGE_MODELS.get(language, LANGUAGE_MODELS["English"])
    processor, model = _load_wav2vec2(checkpoint)

    speech, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    duration_sec = len(speech) / TARGET_SR

    inputs = processor(speech, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    transcript = processor.batch_decode(pred_ids)[0].strip().lower()

    return {"transcript": transcript, "duration_sec": duration_sec, "language": language}


def _filler_count(transcript: str, language: str) -> int:
    fillers = FILLER_WORDS.get(language.lower(), FILLER_WORDS["english"])
    text = " " + " ".join(transcript.lower().split()) + " "
    return sum(text.count(f" {f} ") for f in fillers)


def analyze_filler_words(transcript: str, language: str) -> dict:
    fillers = FILLER_WORDS.get(language.lower(), FILLER_WORDS["english"])
    normalized = [token for token in transcript.lower().split() if token]
    counts = {word: 0 for word in fillers}
    for token in normalized:
        if token in counts:
            counts[token] += 1
    filtered_counts = {k: v for k, v in counts.items() if v > 0}
    return {
        "total": sum(filtered_counts.values()),
        "counts": filtered_counts,
    }


def _pause_stats(audio_path: str, top_db: int = 30) -> dict:
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) < 2:
        return {"pause_count": 0, "avg_pause_sec": 0.0}
    pauses = [
        (intervals[i][0] - intervals[i - 1][1]) / sr
        for i in range(1, len(intervals))
        if (intervals[i][0] - intervals[i - 1][1]) / sr > 0.2
    ]
    return {"pause_count": len(pauses), "avg_pause_sec": float(np.mean(pauses)) if pauses else 0.0}


def analyze_transcript_quality(transcript: str, language: str = "English") -> dict:
    tokens = [token.strip(".,!?;:") for token in transcript.lower().split() if token.strip(".,!?;:")]
    filler_info = analyze_filler_words(transcript, language)

    tokens_to_check = [t for t in tokens if t not in COMMON_WORD_WHITELIST]

    unclear_tokens = []
    for token in tokens_to_check:
        has_vowel = any(char in "aeiou" for char in token)
        is_short_fragment = len(token) <= 4
        if is_short_fragment and token not in COMMON_WORD_WHITELIST:
            unclear_tokens.append(token)
        elif not has_vowel:
            unclear_tokens.append(token)
        elif re.search(r"(.)\1\1", token):
            unclear_tokens.append(token)

    unique_unclear = list(dict.fromkeys(unclear_tokens))
    clarity_score = max(0, 100 - len(unique_unclear) * 18 - max(0, len(tokens) - 6) * 3)
    if unique_unclear:
        message = (
            "This transcript looks short or unclear. Try speaking more slowly, pausing between words, "
            f"and focusing on clearer syllables: {', '.join(unique_unclear)}"
        )
    else:
        message = "This transcript looks clear enough to score. Keep practicing with steady pacing."
    return {
        "unclear_tokens": unique_unclear,
        "clarity_score": clarity_score,
        "message": message,
        "filler_count": filler_info["total"],
    }


def compute_fluency(audio_path: str, transcript: str, duration_sec: float, language: str = "English") -> dict:
    word_count = len(transcript.split())

    MIN_DURATION_SEC = 4.0
    MIN_WORD_COUNT = 4
    if duration_sec < MIN_DURATION_SEC or word_count < MIN_WORD_COUNT:
        return {
            "word_count": word_count,
            "duration_sec": round(duration_sec, 1),
            "words_per_minute": None,
            "filler_count": 0,
            "filler_ratio": 0.0,
            "pause_count": 0,
            "avg_pause_sec": 0.0,
            "fluency_score": None,
            "too_short": True,
        }

    duration_min = max(duration_sec / 60, 1e-6)
    wpm = word_count / duration_min

    fillers = _filler_count(transcript, language)
    filler_ratio = fillers / max(word_count, 1)
    pause_stats = _pause_stats(audio_path)

    quality = analyze_transcript_quality(transcript, language)
    lo, hi = IDEAL_WPM_RANGE
    rate_score = 100 if lo <= wpm <= hi else max(0, 100 - min(abs(wpm - lo), abs(wpm - hi)) * 0.8)
    filler_penalty = min(filler_ratio * 300, 40)
    pause_penalty = min(pause_stats["pause_count"] * 3, 30)
    clarity_penalty = max(0, 30 - quality["clarity_score"] // 2)
    fluency_score = max(0, round(rate_score - filler_penalty - pause_penalty - clarity_penalty))

    return {
        "word_count": word_count,
        "duration_sec": round(duration_sec, 1),
        "words_per_minute": round(wpm, 1),
        "filler_count": fillers,
        "filler_ratio": round(filler_ratio, 3),
        "pause_count": pause_stats["pause_count"],
        "avg_pause_sec": round(pause_stats["avg_pause_sec"], 2),
        "fluency_score": fluency_score,
        "too_short": False,
    }
