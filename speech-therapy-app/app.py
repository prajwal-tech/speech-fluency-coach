import os
import re
import tempfile
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from constants import LANGUAGE_MODELS
from data_utils import get_ollama_models, load_sessions, load_usage, log_chat_usage, log_session
from pronunciation import HAS_TTS, get_language_pronunciation_guide, generate_pronunciation_advice, synthesize_speech
from rag_utils import rag_answer
from speech_utils import analyze_filler_words, analyze_transcript_quality, compute_fluency, transcribe_audio

st.set_page_config(page_title="Speech Therapy AI Suite", layout="wide")

st.sidebar.title("⚙️ Settings")
available_models = get_ollama_models()
if not available_models:
    st.sidebar.warning("Couldn't reach Ollama at localhost:11434. Is it running?")
    available_models = [
        "qwen2.5:1.5b", "llama3.2:1b", "gemma3:1b", "deepseek-r1:1.5b",
        "smollm2:360m", "smollm2:135m", "qwen2.5:0.5b",
    ]

model_choice = st.sidebar.selectbox("Ollama model (used for the RAG chatbot)", available_models)
language_choice = st.sidebar.selectbox("Language", list(LANGUAGE_MODELS.keys()))
client_label = st.sidebar.text_input("Client / session label", value="client-01")
st.sidebar.caption(
    "Runs fully locally: Wav2Vec2 for transcription, sentence-transformers for retrieval, and your local Ollama model for generation."
)

st.title("🗣️ Speech Therapy AI Suite")
tab1, tab2, tab3 = st.tabs(["Transcribe & Fluency", "RAG Training Assistant", "Dashboard"])


def process_audio(audio_bytes: bytes, suffix: str = ".wav"):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        with st.spinner(f"Running Wav2Vec2 ({language_choice})..."):
            result = transcribe_audio(tmp_path, language=language_choice)

        st.markdown("**Transcript**")
        st.write(result["transcript"] or "_(no speech detected)_")

        transcript_quality = analyze_transcript_quality(result["transcript"], language_choice)
        st.caption(transcript_quality["message"])
        if transcript_quality["unclear_tokens"]:
            st.write("Possible unclear tokens: " + ", ".join(transcript_quality["unclear_tokens"]))

        with st.spinner("Computing fluency metrics..."):
            metrics = compute_fluency(tmp_path, result["transcript"], result["duration_sec"], language=language_choice)

        if metrics["too_short"]:
            st.warning(
                f"Clip too short to score fluency ({metrics['duration_sec']}s, {metrics['word_count']} word(s)). "
                "Try recording at least 4 seconds of continuous speech."
            )
            return

        cols = st.columns(4)
        cols[0].metric("Fluency score", f"{metrics['fluency_score']}/100")
        cols[1].metric("Words / min", metrics["words_per_minute"])
        cols[2].metric("Filler words", metrics["filler_count"])
        cols[3].metric("Pauses", metrics["pause_count"])

        filler_analysis = analyze_filler_words(result["transcript"], language_choice)
        if filler_analysis["total"] > 0:
            st.markdown("**Speech habits**")
            st.write(f"You used {filler_analysis['total']} filler word(s).")
            st.write("Common filler words detected: " + ", ".join(f"{word} ({count})" for word, count in filler_analysis["counts"].items()))
        else:
            st.caption("No obvious filler words detected in this sample.")

        guide = get_language_pronunciation_guide(result["transcript"], language_choice)
        st.markdown("**Pronunciation practice**")
        if guide:
            st.write("These are the main spoken words and a simple respelling.")
            for i, item in enumerate(guide):
                c1, c2, c3 = st.columns([3, 4, 1])
                c1.write(item["word"])
                c2.write(item["guide"])
                if c3.button("🔊 Listen", key=f"listen_{i}_{item['word']}"):
                    if HAS_TTS:
                        with st.spinner("Generating audio..."):
                            try:
                                st.audio(synthesize_speech(item["word"]), format="audio/wav")
                            except Exception as e:
                                st.error(f"Voice agent failed: {e}")
                    else:
                        st.warning("Voice agent needs `pyttsx3` installed (pip install pyttsx3).")
            st.caption("Practice the sounds slowly, then connect them smoothly.")
        else:
            st.caption("Pronunciation guide is not available for this phrase yet. Try a simpler word or phrase.")

        log_session({
            "timestamp": datetime.now().isoformat(),
            "client_label": client_label,
            "language": language_choice,
            "clarity_score": transcript_quality["clarity_score"],
            **{k: v for k, v in metrics.items() if k != "too_short"},
        })
        st.success("Session logged to the dashboard.")

    except Exception as e:
        st.error(f"Transcription failed: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


with tab1:
    st.subheader("Record live or upload a session audio clip")
    source = st.radio("Audio source", ["🎙️ Record live", "📁 Upload a file"], horizontal=True)

    if source == "🎙️ Record live":
        recording = st.audio_input("Click to record, click again to stop")
        if recording is not None:
            st.audio(recording)
            if st.button("Transcribe + Score Fluency", type="primary", key="transcribe_live"):
                process_audio(recording.getvalue(), suffix=".wav")
    else:
        audio_file = st.file_uploader("WAV, MP3, M4A, or FLAC", type=["wav", "mp3", "m4a", "flac"])
        if audio_file is not None:
            st.audio(audio_file)
            if st.button("Transcribe + Score Fluency", type="primary", key="transcribe_upload"):
                process_audio(audio_file.read(), suffix=os.path.splitext(audio_file.name)[1] or ".wav")


with tab2:
    st.subheader("RAG Training Assistant")
    st.caption("Ask questions about speech strategies, practice goals, or therapy planning.")

    question = st.text_input("Ask the assistant a question", value="How can I reduce filler words?")
    if st.button("Ask the assistant", key="rag_ask"):
        with st.spinner("Fetching answer..."):
            try:
                rag_response = rag_answer(question, model_choice, top_k=3)
                st.markdown("**Assistant**")
                st.write(rag_response["answer"])
                st.markdown("**Sources**")
                st.write(", ".join(rag_response["sources"]))
                log_chat_usage(question)
            except Exception as e:
                st.error(f"RAG assistant failed: {e}")

    st.markdown("---")
    st.subheader("Pronunciation coach")
    st.caption("Type a word or short phrase and get chat-style guidance on how to pronounce it.")

    practice_text = st.text_input("Word or phrase to practice", value="hello", key="pronounce_text")
    practice_language = st.selectbox(
        "Language", list(LANGUAGE_MODELS.keys()),
        index=list(LANGUAGE_MODELS.keys()).index(language_choice),
        key="pronounce_language",
    )

    if st.button("Ask how to pronounce", key="ask_pronounce"):
        st.session_state["pronunciation_text"] = practice_text
        try:
            guide = get_language_pronunciation_guide(practice_text, practice_language)
            if guide:
                st.session_state["pronunciation_guide"] = guide
                st.session_state["pronunciation_display"] = ""
            else:
                st.session_state["pronunciation_guide"] = []
                st.session_state["pronunciation_display"] = generate_pronunciation_advice(practice_text, practice_language)
        except Exception as e:
            st.error(f"Could not generate guidance: {e}")

    if st.session_state.get("pronunciation_text"):
        st.markdown("**Assistant**")
        if st.session_state.get("pronunciation_guide"):
            def bold_stress(respell: str) -> str:
                return re.sub(r"([A-Z]{2,})", r"**\1**", respell)

            for item in st.session_state["pronunciation_guide"]:
                st.markdown(f"- **{item['word']}**: {bold_stress(item['guide'])}")
            st.markdown("**Practice tips:** say slowly, break into syllables, emphasize the **BOLDED** syllable.")
        else:
            st.code(st.session_state.get("pronunciation_display", ""))

        if HAS_TTS:
            st.markdown("**Audio**")
            voice_choice = st.radio("Voice", ["default", "female", "male"], index=0, horizontal=True, key="pronounce_voice")
            rate_choice = st.slider("Speech rate", min_value=80, max_value=300, value=150, key="pronounce_rate")
            if st.button("🔊 Play full phrase", key="play_full_phrase"):
                try:
                    st.audio(
                        synthesize_speech(
                            st.session_state["pronunciation_text"],
                            voice_gender=(None if voice_choice == "default" else voice_choice),
                            rate=rate_choice,
                        ),
                        format="audio/wav",
                    )
                except Exception as e:
                    st.error(f"Voice agent failed: {e}")

            words = [w for w in st.session_state["pronunciation_text"].split() if w]
            for i, word in enumerate(words):
                col1, col2 = st.columns([6, 1])
                col1.write(word)
                if col2.button("🔊", key=f"play_word_{i}"):
                    try:
                        st.audio(
                            synthesize_speech(word, voice_gender=(None if voice_choice == "default" else voice_choice), rate=rate_choice),
                            format="audio/wav",
                        )
                    except Exception as e:
                        st.error(f"Voice agent failed: {e}")
        else:
            st.warning("Voice agent needs `pyttsx3` installed (pip install pyttsx3) to play audio.")

        st.markdown("**Practice tip**")
        st.write("Say the word slowly, break into syllables, and practice the CAPITALIZED or hyphenated syllable pattern shown above.")


with tab3:
    st.subheader("Fluency trends")
    sessions = load_sessions()

    if sessions.empty:
        st.info("No sessions logged yet — transcribe a clip in the first tab to populate this dashboard.")
    else:
        sessions["clarity_score"] = pd.to_numeric(sessions.get("clarity_score", 0), errors="coerce")
        sessions["filler_ratio"] = pd.to_numeric(sessions.get("filler_ratio", 0), errors="coerce")
        sessions["fluency_score"] = pd.to_numeric(sessions.get("fluency_score", 0), errors="coerce")

        c1, c2, c3 = st.columns(3)
        c1.metric("Sessions logged", len(sessions))
        c2.metric("Avg fluency score", round(sessions["fluency_score"].mean(), 1))
        c3.metric("Avg clarity", round(sessions["clarity_score"].mean(), 1))

        with st.expander("Session highlights"):
            st.write(f"Best clarity: {int(sessions.loc[sessions['clarity_score'].idxmax()]['clarity_score'])}")
            st.write(f"Best fluency: {int(sessions.loc[sessions['fluency_score'].idxmax()]['fluency_score'])}")
            avg_filler = sessions["filler_ratio"].mean()
            if avg_filler < 0.1:
                st.write("Top strength: Low filler use")
                st.write("Top challenge: Keep building clarity and consistency.")
            else:
                st.write("Top strength: Speech rate practice")
                st.write("Top challenge: Reduce filler words and pauses.")

        st.plotly_chart(px.line(sessions, x="timestamp", y="fluency_score", color="client_label", markers=True, title="Fluency score over time"), use_container_width=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(px.bar(sessions, x="timestamp", y="filler_count", color="client_label", title="Filler words per session"), use_container_width=True)
        with col_b:
            st.plotly_chart(px.line(sessions, x="timestamp", y="words_per_minute", color="client_label", markers=True, title="Speech rate (WPM) over time"), use_container_width=True)

        st.markdown("**Raw session log**")
        st.dataframe(sessions.sort_values("timestamp", ascending=False), use_container_width=True)

        sessions_sorted = sessions.sort_values("timestamp").copy()
        sessions_sorted["first_score"] = sessions_sorted.groupby("client_label")["fluency_score"].transform("first")
        sessions_sorted["improvement_vs_first"] = sessions_sorted["fluency_score"] - sessions_sorted["first_score"]

        client_sessions = sessions_sorted[sessions_sorted["client_label"] == client_label]
        if not client_sessions.empty:
            first_score = float(client_sessions["fluency_score"].iloc[0])
            latest_score = float(client_sessions["fluency_score"].iloc[-1])
            c1, c2 = st.columns(2)
            c1.metric(f"{client_label} — first session", first_score)
            c2.metric(f"{client_label} — latest session", latest_score, delta=round(latest_score - first_score, 1))

        st.plotly_chart(
            px.bar(sessions_sorted, x="timestamp", y="improvement_vs_first", color="client_label", title="Fluency change vs. first session").add_hline(y=0, line_dash="dash"),
            use_container_width=True,
        )

        st.markdown("**How long each session took**")
        st.plotly_chart(px.line(sessions_sorted, x="timestamp", y="duration_sec", color="client_label", markers=True, title="Speech duration over time"), use_container_width=True)
        st.plotly_chart(px.scatter(sessions_sorted, x="duration_sec", y="fluency_score", color="client_label", size="word_count", title="Session duration vs. fluency score"), use_container_width=True)

    st.subheader("Training assistant usage")
    usage = load_usage()
    if usage.empty:
        st.info("No chatbot queries logged yet.")
    else:
        usage["date"] = usage["timestamp"].dt.date
        daily = usage.groupby("date").size().reset_index(name="queries")
        st.plotly_chart(px.bar(daily, x="date", y="queries", title="Chatbot queries per day"), use_container_width=True)
        with st.expander("Recent queries"):
            st.dataframe(usage.sort_values("timestamp", ascending=False).head(10), use_container_width=True)
