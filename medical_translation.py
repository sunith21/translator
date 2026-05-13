from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MarianMTModel, MarianTokenizer

HINDI_KEY = "Hindi (हिन्दी)"
_LANGUAGE_ALIASES = {
    "Hindi (हिन्दी)": "Hindi (हिन्दी)",
    "Hindi (à¤¹à¤¿à¤¨à¥à¤¦à¥€)": "Hindi (हिन्दी)",
    "Kannada (ಕನ್ನಡ)": "Kannada (ಕನ್ನಡ)",
    "Kannada (à²•à²¨à³à²¨à²¡)": "Kannada (ಕನ್ನಡ)",
    "Marathi (मराठी)": "Marathi (मराठी)",
    "Marathi (à¤®à¤°à¤¾à¤ à¥€)": "Marathi (मराठी)",
    "Bengali (বাংলা)": "Bengali (বাংলা)",
    "Bengali (à¦¬à¦¾à¦‚à¦²à¦¾)": "Bengali (বাংলা)",
    "Malayalam (മലയാളം)": "Malayalam (മലയാളം)",
    "Malayalam (à´®à´²à´¯à´¾à´³à´‚)": "Malayalam (മലയാളം)",
    "Tamil (தமிழ்)": "Tamil (தமிழ்)",
    "Tamil (à®¤à®®à®¿à®´à¯)": "Tamil (தமிழ்)",
    "Konkani (कोंकणी)": "Konkani (कोंकणी)",
    "Konkani (à¤•à¥‹à¤‚à¤•à¤£à¥€)": "Konkani (कोंकणी)",
}
# Override with NLLB_MODEL_NAME in the environment for quality vs. speed (e.g. facebook/nllb-200-1.3B).
NLLB_MODEL_NAME = os.environ.get(
    "NLLB_MODEL_NAME",
    "facebook/nllb-200-distilled-600M",
)
NMT_ENGINE_NAME = "NLLB-200 offline NMT"
NMT_ENGINE_SUMMARY = (
    "NLLB-200 offline NMT: translates complete sentence meaning locally after model download, "
    "preserving medical terms, dosage values, and grammar context."
)
NLLB_LANGUAGE_CODES = {
    "Hindi (हिन्दी)": "hin_Deva",
    "Kannada (ಕನ್ನಡ)": "kan_Knda",
    "Marathi (मराठी)": "mar_Deva",
    "Bengali (বাংলা)": "ben_Beng",
    "Malayalam (മലയാളം)": "mal_Mlym",
    "Tamil (தமிழ்)": "tam_Taml",
    "Konkani (कोंकणी)": "mar_Deva",
}

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_CACHE: dict[str, dict[str, Any]] = {}
_MODEL_LOCK = threading.Lock()
_SESSION_WARM_LOCK = threading.Lock()
_warmed_session_keys: set[tuple[str, str]] = set()
_TRANSLATION_CACHE: "OrderedDict[object, str]" = OrderedDict()
_CACHE_LOCK = threading.Lock()
_TRANSLATION_CACHE_MAX = 4096
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?।])\s+|\n+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(/])\s+")
_SPACE_BEFORE_CLOSE_RE = re.compile(r"\s+([/)])")

_MEDICAL_TERMS = [
    "CT scan",
    "ECG",
    "EKG",
    "HbA1c",
    "MRI",
    "ORS",
    "SpO2",
    "T3",
    "T4",
    "TSH",
    "ultrasound",
    "X-ray",
    "amlodipine",
    "amoxicillin",
    "atorvastatin",
    "azithromycin",
    "cefixime",
    "cetirizine",
    "creatinine",
    "hemoglobin",
    "ibuprofen",
    "insulin",
    "levocetirizine",
    "metformin",
    "omeprazole",
    "pantoprazole",
    "paracetamol",
    "salbutamol",
    "telmisartan",
]
_MEDICAL_TERM_RE = re.compile(
    r"(?<!\w)(?:"
    + "|".join(sorted((re.escape(term) for term in _MEDICAL_TERMS), key=len, reverse=True))
    + r")(?!\w)",
    re.IGNORECASE,
)
_MEDICAL_PATTERNS = [
    _MEDICAL_TERM_RE,
    re.compile(
        r"(?<!\w)\d+(?:\.\d+)?\s?(?:mcg|mg|g|kg|ml|mL|l|L|IU|iu|units?|mmHg|mmhg|bpm|%|mg/dL|mmol/L)(?!\w)",
        re.IGNORECASE,
    ),
    re.compile(r"(?<!\w)\d{2,3}/\d{2,3}(?!\w)"),
    re.compile(r"(?<!\w)(?:once|twice|thrice)\s+(?:daily|a day)(?!\w)", re.IGNORECASE),
]

if _DEVICE == "cpu":
    try:
        cpu_threads = max(1, min(8, os.cpu_count() or 1))
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(max(1, min(4, cpu_threads)))
    except RuntimeError:
        pass


def _cache_get(key: object) -> str | None:
    with _CACHE_LOCK:
        value = _TRANSLATION_CACHE.get(key)
        if value is not None:
            _TRANSLATION_CACHE.move_to_end(key)
        return value


def _cache_set(key: object, value: str) -> None:
    with _CACHE_LOCK:
        _TRANSLATION_CACHE[key] = value
        _TRANSLATION_CACHE.move_to_end(key)
        while len(_TRANSLATION_CACHE) > _TRANSLATION_CACHE_MAX:
            _TRANSLATION_CACHE.popitem(last=False)


def _prepare_model(model):
    model.eval()
    if next(model.parameters()).device.type != _DEVICE:
        model.to(_DEVICE)
    return model


def _get_marian(model_name: str):
    with _MODEL_LOCK:
        if model_name not in _MODEL_CACHE:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            _MODEL_CACHE[model_name] = {
                "tokenizer": tokenizer,
                "model": _prepare_model(model),
            }
    return _MODEL_CACHE[model_name]["tokenizer"], _MODEL_CACHE[model_name]["model"]


def _get_nllb():
    with _MODEL_LOCK:
        if NLLB_MODEL_NAME not in _MODEL_CACHE:
            tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME, use_fast=False)
            model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME)
            _MODEL_CACHE[NLLB_MODEL_NAME] = {
                "tokenizer": tokenizer,
                "model": _prepare_model(model),
            }
    return _MODEL_CACHE[NLLB_MODEL_NAME]["tokenizer"], _MODEL_CACHE[NLLB_MODEL_NAME]["model"]


def _normalize_text(text: str) -> str:
    cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    return cleaned


def _split_long_fragment(fragment: str, limit: int) -> list[str]:
    if len(fragment) <= limit:
        return [fragment]

    pieces: list[str] = []
    remaining = fragment
    while len(remaining) > limit:
        cut = remaining.rfind(" ", 0, limit + 1)
        if cut <= max(20, limit // 3):
            cut = remaining.find(" ", limit)
        if cut == -1:
            cut = limit
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _chunk_text(text: str, limit: int = 260) -> list[str]:
    """Build sentence-level chunks so the model translates meaning, not isolated words."""
    normalized = _normalize_text(text)
    if not normalized:
        return []

    fragments = [part.strip() for part in _SENTENCE_BREAK_RE.split(normalized) if part.strip()]
    if not fragments:
        return [normalized]

    chunks: list[str] = []
    current = ""
    for fragment in fragments:
        for piece in _split_long_fragment(fragment, limit):
            if not current:
                current = piece
            elif len(current) + 1 + len(piece) <= limit:
                current = f"{current} {piece}"
            else:
                chunks.append(current)
                current = piece
    if current:
        chunks.append(current)
    return chunks


def _protect_medical_terms(text: str) -> tuple[str, list[str]]:
    replacements: list[str] = []

    def replace(match: re.Match[str]) -> str:
        replacements.append(match.group(0))
        return f" XMEDTERM{len(replacements) - 1}X "

    protected = text
    for pattern in _MEDICAL_PATTERNS:
        protected = pattern.sub(replace, protected)
    return protected, replacements


def _restore_medical_terms(text: str, replacements: list[str]) -> str:
    restored = text
    for index, original in enumerate(replacements):
        restored = restored.replace(f"XMEDTERM{index}X", original)
    restored = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", restored)
    restored = _SPACE_AFTER_OPEN_RE.sub(r"\1", restored)
    restored = _SPACE_BEFORE_CLOSE_RE.sub(r"\1", restored)
    return restored.strip()


def _max_input_length(tokenizer) -> int:
    model_max_length = getattr(tokenizer, "model_max_length", 256)
    if not isinstance(model_max_length, int) or model_max_length <= 0:
        return 256
    return min(model_max_length, 384)


def _generate_batch(tokenizer, model, inputs, forced_bos_token_id: int | None = None) -> list[str]:
    encoded = tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=_max_input_length(tokenizer),
    )
    encoded = encoded.to(_DEVICE)

    generation_kwargs = {
        "max_new_tokens": 256,
        "num_beams": max(1, int(os.environ.get("NLLB_NUM_BEAMS", "3"))),
        "early_stopping": True,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.05,
    }
    if forced_bos_token_id is not None:
        generation_kwargs["forced_bos_token_id"] = forced_bos_token_id

    with torch.inference_mode():
        output_tokens = model.generate(**encoded, **generation_kwargs)
    return [text.strip() for text in tokenizer.batch_decode(output_tokens, skip_special_tokens=True)]


def _translate_chunk_batch(namespace: object, chunks: list[str], batch_translate) -> list[str]:
    results: list[str | None] = [None] * len(chunks)
    pending_chunks: list[str] = []
    pending_indexes: list[int] = []

    for index, chunk in enumerate(chunks):
        cache_key = (namespace, chunk)
        cached = _cache_get(cache_key)
        if cached is None:
            pending_chunks.append(chunk)
            pending_indexes.append(index)
        else:
            results[index] = cached

    if pending_chunks:
        translated = batch_translate(pending_chunks)
        for index, value in zip(pending_indexes, translated):
            results[index] = value
            _cache_set((namespace, chunks[index]), value)

    return [value or "" for value in results]


def _translate_marian_chunks(chunks: list[str], model_name: str) -> list[str]:
    tokenizer, model = _get_marian(model_name)
    namespace = ("marian", model_name)
    return _translate_chunk_batch(namespace, chunks, lambda values: _generate_batch(tokenizer, model, values))


def _translate_nllb_chunks(chunks: list[str], src_lang: str, tgt_lang: str) -> list[str]:
    tokenizer, model = _get_nllb()
    tokenizer.src_lang = src_lang
    bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    namespace = ("nllb", src_lang, tgt_lang)
    return _translate_chunk_batch(
        namespace,
        chunks,
        lambda values: _generate_batch(tokenizer, model, values, forced_bos_token_id=bos_token_id),
    )


def get_nllb_code(lang_key: str) -> str:
    normalized_lang = _LANGUAGE_ALIASES.get(lang_key, lang_key)
    return NLLB_LANGUAGE_CODES.get(normalized_lang, "hin_Deva")


def describe_translation_engine(lang_key: str | None = None) -> str:
    if lang_key:
        return f"{NMT_ENGINE_NAME} - NLLB multilingual sentence model"
    return NMT_ENGINE_SUMMARY


def warm_translation_models(direction: str, lang_key: str) -> None:
    _get_nllb()


def ensure_session_models(doctor_lang_key: str, patient_lang_key: str) -> None:
    """Load NLLB-200 once per unique doctor/patient language pair."""
    dk = _LANGUAGE_ALIASES.get(doctor_lang_key, doctor_lang_key)
    pk = _LANGUAGE_ALIASES.get(patient_lang_key, patient_lang_key)
    key = (dk, pk)
    with _SESSION_WARM_LOCK:
        if key in _warmed_session_keys:
            return
        _warmed_session_keys.add(key)
    _get_nllb()


def translate_medical_text(text: str, direction: str, lang_key: str) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        return ""
    normalized_lang = _LANGUAGE_ALIASES.get(lang_key, lang_key)

    cache_key = ("full", normalized, direction, normalized_lang)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    protected_text, replacements = _protect_medical_terms(normalized)
    chunks = _chunk_text(protected_text)

    if direction == "en_to_native":
        translated_chunks = _translate_nllb_chunks(chunks, "eng_Latn", get_nllb_code(normalized_lang))
    else:
        translated_chunks = _translate_nllb_chunks(chunks, get_nllb_code(normalized_lang), "eng_Latn")

    translated = _restore_medical_terms(" ".join(chunk for chunk in translated_chunks if chunk), replacements)
    _cache_set(cache_key, translated)
    return translated


def translate_medical_native_pair(text: str, source_lang_key: str, target_lang_key: str) -> str:
    """
    Single NLLB pass between two consultation languages (no English pivot).
    Improves accuracy and halves MT latency vs. native→English→native.
    """
    if source_lang_key == "English" or target_lang_key == "English":
        raise ValueError(
            "translate_medical_native_pair does not support English; use translate_medical_text."
        )
    normalized = _normalize_text(text)
    if not normalized:
        return ""
    src_key = _LANGUAGE_ALIASES.get(source_lang_key, source_lang_key)
    tgt_key = _LANGUAGE_ALIASES.get(target_lang_key, target_lang_key)
    if src_key == tgt_key:
        return normalized

    cache_key = ("native_pair", normalized, src_key, tgt_key)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    protected_text, replacements = _protect_medical_terms(normalized)
    chunks = _chunk_text(protected_text)
    src_code = get_nllb_code(src_key)
    tgt_code = get_nllb_code(tgt_key)
    translated_chunks = _translate_nllb_chunks(chunks, src_code, tgt_code)
    translated = _restore_medical_terms(" ".join(c for c in translated_chunks if c), replacements)
    _cache_set(cache_key, translated)
    return translated
