import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SESSIONS_CSV = os.path.join(DATA_DIR, "sessions.csv")
USAGE_CSV = os.path.join(DATA_DIR, "chat_usage.csv")
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
TARGET_SR = 16_000
CHUNK_WORDS = 120

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

FILLER_WORDS = {
    "english": {"um", "uh", "erm", "like", "you know", "so", "actually", "basically", "i mean", "ah"},
    "spanish": {"eh", "este", "o sea", "bueno", "pues"},
    "french": {"euh", "ben", "genre", "quoi", "voila"},
    "german": {"äh", "ähm", "also", "halt", "sozusagen"},
}
IDEAL_WPM_RANGE = (110, 160)
COMMON_WORD_WHITELIST = {
    "a", "an", "the", "and", "or", "but", "on", "in", "up", "down",
    "how", "hows", "hello", "things", "going", "is", "are",
    "it", "this", "that", "to", "of", "for", "with", "as", "at",
    "by", "from", "be", "you", "i", "we", "they", "he", "she",
    "do", "does", "did", "say",
}
LANGUAGE_SPELLING_RULES = {
    "Spanish": [
        ("qu", "k"), ("ll", "y"), ("ñ", "ny"), ("ch", "ch"), ("j", "h"),
        ("gue", "ge"), ("gui", "gi"), ("que", "ke"), ("qui", "ki"),
        ("z", "s"), ("c", "k"), ("v", "b"), ("y", "y"),
    ],
    "French": [
        ("eau", "oh"), ("au", "oh"), ("ou", "oo"), ("oi", "wah"),
        ("an", "ahn"), ("en", "ahn"), ("on", "ohn"), ("gn", "ny"),
        ("ill", "ee"), ("ph", "f"), ("ch", "sh"), ("qu", "k"),
    ],
    "German": [
        ("sch", "sh"), ("ch", "kh"), ("ei", "eye"), ("ie", "ee"),
        ("eu", "oy"), ("ä", "eh"), ("ö", "er"), ("ü", "oo"),
        ("j", "y"), ("w", "v"), ("z", "ts"),
    ],
    "Italian": [
        ("ch", "k"), ("gh", "g"), ("gn", "ny"), ("gli", "ly"),
        ("ci", "chee"), ("ce", "che"), ("ge", "je"), ("gi", "jee"),
        ("sc", "sh"),
    ],
    "Portuguese": [
        ("lh", "ly"), ("nh", "ny"), ("ch", "sh"), ("qu", "k"),
        ("gu", "g"), ("ão", "aw"), ("ã", "aw"), ("ç", "s"),
        ("ei", "ay"), ("eu", "eh-oo"),
    ],
}
LANGUAGE_VOWELS = {
    "Spanish": "aeiouáéíóúü",
    "French": "aeiouyàâçéèêëîïôûùüÿ",
    "German": "aeiouyäöü",
    "Italian": "aeiou",
    "Portuguese": "aeiouáâãàçéêíóôõú",
}
ARPABET_SIMPLE = {
    "AA": "ah", "AE": "a", "AH": "uh", "AO": "aw", "AW": "ow", "AY": "eye",
    "B": "b", "CH": "ch", "D": "d", "DH": "th", "EH": "eh", "ER": "er",
    "EY": "ay", "F": "f", "G": "g", "HH": "h", "IH": "ih", "IY": "ee",
    "JH": "j", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ng",
    "OW": "oh", "OY": "oy", "P": "p", "R": "r", "S": "s", "SH": "sh",
    "T": "t", "TH": "th", "UH": "uh", "UW": "oo", "V": "v", "W": "w",
    "Y": "y", "Z": "z", "ZH": "zh",
}
