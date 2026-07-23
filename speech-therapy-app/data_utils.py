import os
from datetime import datetime

import pandas as pd
import requests

from constants import DATA_DIR, OLLAMA_TAGS_URL, SESSIONS_CSV, USAGE_CSV


def get_ollama_models():
    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=3)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


def log_session(row: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(SESSIONS_CSV, mode="a", header=not os.path.exists(SESSIONS_CSV), index=False)


def log_chat_usage(query: str = None):
    os.makedirs(DATA_DIR, exist_ok=True)
    row = pd.DataFrame([{"timestamp": datetime.now().isoformat(), "query": query or ""}])
    row.to_csv(USAGE_CSV, mode="a", header=not os.path.exists(USAGE_CSV), index=False)


def load_sessions():
    if os.path.exists(SESSIONS_CSV):
        return pd.read_csv(SESSIONS_CSV, parse_dates=["timestamp"])
    return pd.DataFrame(columns=[
        "timestamp", "client_label", "language", "word_count", "duration_sec",
        "words_per_minute", "filler_count", "filler_ratio", "pause_count",
        "avg_pause_sec", "fluency_score", "clarity_score",
    ])


def load_usage():
    if os.path.exists(USAGE_CSV):
        try:
            return pd.read_csv(USAGE_CSV, parse_dates=["timestamp"])
        except pd.errors.ParserError:
            with open(USAGE_CSV, "r", encoding="utf-8", errors="replace") as f:
                raw_lines = [line.rstrip("\n\r") for line in f if line.strip()]

            if not raw_lines:
                return pd.DataFrame(columns=["timestamp", "query"])

            header = raw_lines[0].strip()
            if header == "timestamp":
                raw_lines[0] = "timestamp,query"
            elif not header.startswith("timestamp,"):
                raw_lines.insert(0, "timestamp,query")

            repaired = [raw_lines[0]]
            for line in raw_lines[1:]:
                if "," in line:
                    repaired.append(line)
                else:
                    repaired.append(f"{line},")

            with open(USAGE_CSV, "w", encoding="utf-8", newline="") as f:
                f.write("\n".join(repaired) + "\n")

            return pd.read_csv(USAGE_CSV, parse_dates=["timestamp"])
    return pd.DataFrame(columns=["timestamp", "query"])
