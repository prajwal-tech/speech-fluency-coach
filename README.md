# Speech Fluency Coach

A local, multilingual speech fluency & pronunciation coaching app —
transcription, fluency tracking, pronunciation practice with a voice
agent, and a RAG chatbot for therapy technique guidance. Runs fully
offline: transcription via Wav2Vec2, retrieval via sentence-transformers,
and generation via your local Ollama model. No cloud APIs, no client
audio or data ever leaves your machine.

## Features

- **Multilingual speech-to-text** — Wav2Vec2.0 (XLSR-53), 12 languages, live mic recording or file upload
- **Fluency scoring** — heuristic score from speech rate, filler-word ratio, and pause detection, tracked over sessions
- **Pronunciation coaching** — phonetic breakdown (CMU dictionary) side-by-side with your transcript, plus an offline text-to-speech voice agent to hear correct pronunciation
- **RAG training assistant** — chatbot grounded in an embedded speech-therapy knowledge base, answered by your local Ollama model
- **Dashboard** — fluency trends, filler/pause charts, and improvement-over-time tracking per client

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- At least one small chat model pulled, e.g. `ollama pull qwen2.5:1.5b`

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first transcription in a given language downloads that language's
Wav2Vec2 checkpoint once, then it's cached locally.

## Run

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## Usage

- **Transcribe & Fluency** — record live or upload a clip, get a transcript, fluency score, and pronunciation guide with listen buttons
- **RAG Training Assistant** — ask therapy technique questions; answers cite their source when grounded in the knowledge base
- **Dashboard** — view fluency trends, filler/pause charts, and improvement vs. each client's first session

## Measuring accuracy

`evaluate_accuracy.py` computes Word Error Rate (WER) and accuracy
against a manifest of audio clips + known-correct transcripts:

```bash
python evaluate_accuracy.py --manifest manifest_example.csv
```

Fill in `manifest_example.csv` with your own clips and their correct
transcripts to get a real, reproducible accuracy number.

## Notes & limitations

- The fluency score is a **heuristic demo metric**, not a clinically validated assessment
- Pronunciation guide and voice agent are English-only (CMU dictionary has no multilingual equivalent)
- Wav2Vec2 accuracy varies by language and audio quality; English is the strongest of the included checkpoints
