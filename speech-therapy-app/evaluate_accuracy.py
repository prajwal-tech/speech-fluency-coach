

import argparse
import csv
import functools

import jiwer
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

LANGUAGE_MODELS = {
    "English": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    "Spanish": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "French": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "German": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    "Italian": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
    "Portuguese": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "Dutch": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "Hindi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "Arabic": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "Russian": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "Chinese": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "Japanese": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
}
TARGET_SR = 16_000


@functools.lru_cache(maxsize=4)
def _load_model(checkpoint: str):
    processor = Wav2Vec2Processor.from_pretrained(checkpoint)
    model = Wav2Vec2ForCTC.from_pretrained(checkpoint)
    model.eval()
    return processor, model


def transcribe(audio_path: str, language: str = "English") -> str:
    checkpoint = LANGUAGE_MODELS.get(language, LANGUAGE_MODELS["English"])
    processor, model = _load_model(checkpoint)
    speech, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    inputs = processor(speech, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0].strip().lower()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Wav2Vec2 transcription accuracy (WER-based).")
    parser.add_argument("--manifest", required=True, help="Path to manifest CSV")
    args = parser.parse_args()

    references, hypotheses, rows = [], [], []

    with open(args.manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            language = row.get("language") or "English"
            reference = row["reference_transcript"].strip().lower()
            print(f"Transcribing {row['audio_path']}...")
            hypothesis = transcribe(row["audio_path"], language=language)

            clip_wer = jiwer.wer(reference, hypothesis)
            references.append(reference)
            hypotheses.append(hypothesis)
            rows.append((row["audio_path"], reference, hypothesis, clip_wer))

            print(f"  reference:  {reference}")
            print(f"  hypothesis: {hypothesis}")
            print(f"  WER: {clip_wer:.2%}\n")

    if not rows:
        print("No rows found in manifest.")
        return

    overall_wer = jiwer.wer(references, hypotheses)
    accuracy = 1 - overall_wer

    print("=" * 40)
    print(f"Clips evaluated: {len(rows)}")
    print(f"Overall WER:      {overall_wer:.2%}")
    print(f"Overall accuracy: {accuracy:.2%}")
    print("=" * 40)


if __name__ == "__main__":
    main()
