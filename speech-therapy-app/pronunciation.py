import os
import re
import tempfile

try:
    import pronouncing
    HAS_PRONOUNCING = True
except ImportError:
    pronouncing = None
    HAS_PRONOUNCING = False

try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    pyttsx3 = None
    HAS_TTS = False

from constants import ARPABET_SIMPLE, LANGUAGE_SPELLING_RULES, LANGUAGE_VOWELS


def _respell(arpabet_phones: str) -> str:
    syllables = []
    for phone in arpabet_phones.split():
        stress = phone[-1] if phone and phone[-1].isdigit() else None
        base = phone.rstrip("012")
        simple = ARPABET_SIMPLE.get(base, base.lower())
        if stress == "1":
            simple = simple.upper()
        syllables.append(simple)
    return "-".join(syllables)


def _normalize_word(word: str) -> str:
    return "".join(ch for ch in word.lower() if ch.isalpha() or ch in "áéíóúüàâçèêëîïôõùûÿñ")


def _respell_with_rules(word: str, language: str) -> str:
    base = _normalize_word(word)
    rules = LANGUAGE_SPELLING_RULES.get(language, [])
    for src, repl in rules:
        base = base.replace(src, repl)
    vowels = LANGUAGE_VOWELS.get(language, "aeiouy")
    parts = re.findall(fr"[{vowels}]+[^{vowels}]*|[^{vowels}]+", base)
    return "-".join(part for part in parts if part)


def get_language_pronunciation_guide(transcript: str, language: str) -> list:
    guide, seen = [], set()
    for raw in transcript.split():
        word = _normalize_word(raw)
        if not word or word in seen:
            continue
        seen.add(word)
        if language == "English" and HAS_PRONOUNCING:
            phones_list = pronouncing.phones_for_word(word)
            if phones_list:
                guide.append({"word": word, "guide": _respell(phones_list[0])})
                continue
        if language in LANGUAGE_SPELLING_RULES:
            guide.append({"word": word, "guide": _respell_with_rules(word, language)})
        else:
            parts = re.findall(r"[aeiouy]+[^aeiouy]*|[^aeiouy]+", word.lower())
            guide.append({"word": word, "guide": "-".join(parts)})
    return guide


def generate_pronunciation_advice(text: str, language: str = "English") -> str:
    words = [w for w in text.split() if w]
    if not words:
        return "Please enter a word or short phrase to get pronunciation guidance."

    guide = get_language_pronunciation_guide(text, language)
    if guide:
        lines = [f"Here's how to say each word in {language}:"]
        for item in guide:
            lines.append(f"- {item['word']}: {item['guide']}")
        lines.append("")
        if language == "English":
            lines.append("Practice tips: say slowly, break into syllables, and emphasize the CAPITALIZED syllable.")
        else:
            lines.append("Practice tip: keep vowel sounds clean and consistent for this language.")
        return "\n".join(lines)

    lines = ["Pronunciation (heuristic):"]
    for w in words:
        parts = re.findall(r"[aeiouy]+[^aeiouy]*|[^aeiouy]+", w.lower())
        lines.append(f"- {w}: {'-'.join(parts)}")
    lines.append("")
    lines.append("Practice tip: say each syllable slowly, then link them together. Use a steady rhythm.")
    return "\n".join(lines)


def synthesize_speech(text: str, voice_gender: str = None, rate: int = None) -> bytes:
    if not HAS_TTS:
        raise RuntimeError("pyttsx3 is required for speech synthesis")
    engine = pyttsx3.init()

    try:
        if voice_gender:
            voices = engine.getProperty("voices")
            target = None
            gender = voice_gender.lower()
            for v in voices:
                name = (v.name or "").lower()
                vid = (v.id or "").lower()
                if gender == "female" and ("female" in name or "f" in vid or "woman" in name):
                    target = v.id
                    break
                if gender == "male" and ("male" in name or "m" in vid or "man" in name):
                    target = v.id
                    break
            if target:
                engine.setProperty("voice", target)
    except Exception:
        pass

    try:
        if rate is not None:
            engine.setProperty("rate", int(rate))
    except Exception:
        pass

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name
    engine.save_to_file(text, tmp_path)
    engine.runAndWait()
    engine.stop()
    with open(tmp_path, "rb") as f:
        audio_bytes = f.read()
    try:
        os.unlink(tmp_path)
    except Exception:
        pass
    return audio_bytes
