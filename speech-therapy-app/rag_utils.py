import numpy as np
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

from constants import CHUNK_WORDS, OLLAMA_CHAT_URL

KNOWLEDGE_BASE = {
    "communication_strategies.md": """
Echolalia (repeating words or phrases heard previously) is often functional
communication rather than "just noise." Immediate echolalia can be a
processing strategy; delayed echolalia often carries meaning tied to the
original context it was learned in.

Augmentative and Alternative Communication (AAC) includes picture exchange
systems (PECS), speech-generating devices, and sign systems. Introducing AAC
early does not delay verbal speech development.

Visual schedules, first-then boards, and social stories reduce uncertainty
and support transitions. Pair visuals with consistent verbal language.

Prompting hierarchy, least to most intrusive: natural cue/wait time,
gestural prompt, verbal prompt, model prompt, physical prompt. Fade prompts
systematically to build independence rather than prompt-dependence.
""",
    "fluency_techniques.md": """
Rate control: slowed, stretched speech ("easy onset") reduces tension-based
disfluencies. Practice slow first, then increase toward natural pace.

Reducing filler words: high filler use often correlates with word-finding
effort or anxiety about pausing. Teach comfortable pausing as an alternative
to filling silence — it reads as more fluent to listeners.

Pausing and phrasing: encourage pausing at natural syntactic boundaries
(end of clauses/sentences) rather than mid-phrase.

Useful session-to-session metrics: words per minute, filler ratio, pause
count/duration. Track trends across multiple sessions, not single-session
variability.
""",
    "session_planning.md": """
Embedding a client's special interests into therapy targets increases
engagement and generalization compared to generic drill materials.

Check sensory load before starting (lighting, noise, seating) — a short
regulation activity before language work often improves session output.

Review the last 2-3 sessions' metrics before planning the next one. If
scores trend down, consider difficulty, session length, or reinforcement
before assuming skill regression.

Brief, concrete take-home strategies (one or two items) are more likely to
be practiced between sessions than a long home program.
""",
}


def _chunk_text(text: str, source: str):
    words = text.split()
    return [
        {"text": " ".join(words[i : i + CHUNK_WORDS]), "source": source}
        for i in range(0, len(words), CHUNK_WORDS)
        if words[i : i + CHUNK_WORDS]
    ]


@st.cache_resource
def _get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def _build_index():
    chunks = []
    for source, text in KNOWLEDGE_BASE.items():
        chunks.extend(_chunk_text(text, source))
    embedder = _get_embedder()
    embeddings = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True)
    return chunks, embeddings


def retrieve(query: str, top_k: int = 3):
    chunks, embeddings = _build_index()
    if len(chunks) == 0:
        return []
    embedder = _get_embedder()
    q_emb = embedder.encode([query], normalize_embeddings=True)[0]
    sims = embeddings @ q_emb
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [{"text": chunks[i]["text"], "source": chunks[i]["source"], "score": float(sims[i])} for i in top_idx]


def rag_answer(query: str, model: str, top_k: int = 3) -> dict:
    hits = retrieve(query, top_k=top_k)
    context_lines = []
    for h in hits:
        context_lines.append(f"[{h['source']}]")
        context_lines.append(h['text'])
    context = "\n\n".join(context_lines)
    system_prompt = (
        "You are a training-support assistant for speech-language therapists "
        "working with autistic clients. Answer using the provided context when "
        "relevant. If the context doesn't cover the question, say so and answer "
        "from general best-practice knowledge. Be concise and practical."
    )
    resp = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    response_json = resp.json()
    answer = response_json["message"]["content"]
    sources = [chunk["source"] for chunk in hits]
    return {"answer": answer, "sources": sources, "query": query}
