import os
import json
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
import requests
import whisper
import torch
import numpy as np
import scipy.io.wavfile as wav
from transformers import MarianMTModel, MarianTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
from fpdf import FPDF
from pydub import AudioSegment

# ─────────────────────────────────────────────
#  Lazy Model Cache
# ─────────────────────────────────────────────
_MODEL_CACHE = {}

def get_whisper(size="base"):
    if f"whisper_{size}" not in _MODEL_CACHE:
        _MODEL_CACHE[f"whisper_{size}"] = whisper.load_model(size)
    return _MODEL_CACHE[f"whisper_{size}"]

def get_marian(model_name):
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = {
            "tokenizer": MarianTokenizer.from_pretrained(model_name),
            "model": MarianMTModel.from_pretrained(model_name),
        }
    return _MODEL_CACHE[model_name]

def get_nllb():
    model_name = "facebook/nllb-200-distilled-600M"
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = {
            "tokenizer": AutoTokenizer.from_pretrained(model_name, use_fast=False),
            "model": AutoModelForSeq2SeqLM.from_pretrained(model_name),
        }
    return _MODEL_CACHE[model_name]

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
BASE_DIR = Path("data")
BASE_DIR.mkdir(exist_ok=True)
PATIENTS_DIR = BASE_DIR / "patients"
PATIENTS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
#  Patient Manager
# ─────────────────────────────────────────────
class PatientManager:
    def __init__(self):
        self.db_path = BASE_DIR / "hospital_data.db"
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_visit TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Migrate existing databases that lack the email column
            try:
                conn.execute("ALTER TABLE patients ADD COLUMN email TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass  # Column already exists

    def add_patient(self, patient_id, name, email=""):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO patients (id, name, email, last_visit) VALUES (?, ?, ?, ?)",
                (patient_id, name, email, datetime.now())
            )
        p_dir = PATIENTS_DIR / patient_id
        p_dir.mkdir(exist_ok=True)
        (p_dir / "recordings").mkdir(exist_ok=True)
        log_path = p_dir / "transcripts.json"
        if not log_path.exists():
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def get_patient(self, patient_id):
        with sqlite3.connect(self.db_path) as conn:
            res = conn.execute(
                "SELECT id, name, email, last_visit FROM patients WHERE id = ?",
                (patient_id,)
            ).fetchone()
            if res:
                return {"id": res[0], "name": res[1], "email": res[2] or "", "last_visit": res[3]}
        return None

    def search_patients(self, query):
        with sqlite3.connect(self.db_path) as conn:
            res = conn.execute(
                "SELECT id, name, email FROM patients WHERE id LIKE ? OR name LIKE ? LIMIT 10",
                (f"%{query}%", f"%{query}%")
            ).fetchall()
            return [{"id": r[0], "name": r[1], "email": r[2] or ""} for r in res]

    def get_recent_patients(self):
        with sqlite3.connect(self.db_path) as conn:
            res = conn.execute(
                "SELECT id, name, email, last_visit FROM patients ORDER BY last_visit DESC LIMIT 10"
            ).fetchall()
            return [{"id": r[0], "name": r[1], "email": r[2] or "", "last_visit": r[3]} for r in res]

    def update_last_visit(self, patient_id):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE patients SET last_visit = ? WHERE id = ?", (datetime.now(), patient_id))

# ─────────────────────────────────────────────
#  Email Service
# ─────────────────────────────────────────────
class EmailService:
    """Sends PDF consultation reports to the patient's email via SMTP.
    Configure via environment variables:
        SMTP_HOST   (default: smtp.gmail.com)
        SMTP_PORT   (default: 587)
        SMTP_USER   - your Gmail / SMTP account address
        SMTP_PASS   - Gmail App Password (NOT your regular password)
        SMTP_FROM   - display name + address, e.g. 'Clinic AI <you@gmail.com>'
    """

    @staticmethod
    def send_report(to_email: str, patient_name: str, pdf_path: str) -> dict:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text   import MIMEText
        from email.mime.base   import MIMEBase
        from email             import encoders

        smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.environ.get("SMTP_PORT", 587))
        smtp_user = os.environ.get("SMTP_USER", "")
        smtp_pass = os.environ.get("SMTP_PASS", "")
        smtp_from = os.environ.get("SMTP_FROM", smtp_user)

        if not smtp_user or not smtp_pass:
            raise ValueError(
                "SMTP_USER and SMTP_PASS must be set in the .env file. "
                "For Gmail, generate an App Password at myaccount.google.com/apppasswords."
            )

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        msg = MIMEMultipart()
        msg["From"]    = smtp_from
        msg["To"]      = to_email
        msg["Subject"] = f"Your Medical Consultation Report — {patient_name}"

        body = f"""Dear {patient_name},

Please find attached your medical consultation report from your recent visit.

This report contains a transcript of your consultation along with translations 
to assist you in understanding the discussion with your doctor.

If you have any questions, please contact the clinic directly.

Warm regards,
Clinic AI | Multilingual Medical Assistant"""

        msg.attach(MIMEText(body, "plain", "utf-8"))

        with open(pdf_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        filename = os.path.basename(pdf_path)
        part.add_header("Content-Disposition", f'attachment; filename="{filename}"')
        msg.attach(part)

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_email, msg.as_string())

        print(f"[EmailService] Report sent to {to_email}")
        return {"status": "sent", "to": to_email}

# ─────────────────────────────────────────────
#  AI Service
# ─────────────────────────────────────────────
class AIService:
    SARVAM_STT_URL  = "https://api.sarvam.ai/speech-to-text"
    SARVAM_TTS_URL  = "https://api.sarvam.ai/text-to-speech"
    SARVAM_CHAT_URL = "https://api.sarvam.ai/v1/chat/completions"

    _CORRECTION_SYSTEM_PROMPT = """You are an advanced medical transcript correction assistant.

Your task is to correct speech-to-text transcription errors in a multilingual doctor-patient conversation while preserving the ORIGINAL medical meaning exactly.

Rules:
- Do NOT summarize.
- Do NOT invent symptoms, diagnoses, or medications.
- Do NOT add medical assumptions.
- Correct only probable transcription mistakes.
- Preserve patient intent and tone.
- Keep medical terminology accurate.
- Expand unclear phrases only when highly certain from context.
- Maintain proper punctuation and sentence structure.
- Preserve all numbers, dosages, durations, and measurements carefully.
- If a word is uncertain, keep the original instead of hallucinating.

Context:
This transcript may contain:
- medical symptoms
- medication names
- disease names
- body parts
- multilingual accents
- Indian English pronunciation
- casual patient wording

Common medical vocabulary includes:
diabetes, hypertension, asthma, migraine, fever, nausea,
paracetamol, ibuprofen, insulin, blood pressure, chest pain,
infection, allergy, dizziness, cough, cold, weakness,
stomach pain, headache, breathing difficulty, swelling

Return ONLY the corrected transcript text with no extra commentary."""

    @staticmethod
    def correct_transcript(text: str) -> str:
        """Send STT text through Sarvam's LLM for medical transcription correction."""
        api_key = os.environ.get("SARVAM_API_KEY", "")
        if not api_key:
            print("[correct_transcript] No SARVAM_API_KEY — skipping correction.")
            return text
        try:
            payload = {
                "model": "sarvam-m",
                "messages": [
                    {"role": "system", "content": AIService._CORRECTION_SYSTEM_PROMPT},
                    {"role": "user",   "content": f'"""\n{text}\n"""'}
                ],
                "temperature": 0.1,
                "max_tokens": 512,
            }
            resp = requests.post(
                AIService.SARVAM_CHAT_URL,
                headers={
                    "api-subscription-key": api_key,
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=20
            )
            resp.raise_for_status()
            corrected = resp.json()["choices"][0]["message"]["content"].strip()
            # Safety: if the LLM returns empty or something wildly longer, keep original
            if not corrected or len(corrected) > len(text) * 3:
                return text
            print(f"[correct_transcript] '{text}' → '{corrected}'")
            return corrected
        except Exception as e:
            print(f"[correct_transcript] failed: {e} — using original text")
            return text


    @staticmethod
    def translate(text, direction, lang_key):
        try:
            if direction == "en_to_native":
                if lang_key == "Hindi (हिन्दी)":
                    res = AIService._translate_marian(text, "Helsinki-NLP/opus-mt-en-hi")
                else:
                    tgt = AIService._get_nllb_code(lang_key)
                    res = AIService._translate_nllb(text, "eng_Latn", tgt)
            else:
                if lang_key == "Hindi (हिन्दी)":
                    res = AIService._translate_marian(text, "Helsinki-NLP/opus-mt-hi-en")
                else:
                    src = AIService._get_nllb_code(lang_key)
                    res = AIService._translate_nllb(text, src, "eng_Latn")
            return res
        except Exception as e:
            return f"Error: {str(e)}"

    @staticmethod
    def _translate_marian(text, model_name):
        cache = get_marian(model_name)
        inputs = cache["tokenizer"]([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            tokens = cache["model"].generate(**inputs, max_length=512)
        return cache["tokenizer"].decode(tokens[0], skip_special_tokens=True)

    @staticmethod
    def _translate_nllb(text, src, tgt):
        cache = get_nllb()
        cache["tokenizer"].src_lang = src
        inputs = cache["tokenizer"](text, return_tensors="pt", truncation=True, max_length=512)
        bos_id = cache["tokenizer"].convert_tokens_to_ids(tgt)
        with torch.no_grad():
            tokens = cache["model"].generate(**inputs, forced_bos_token_id=bos_id, max_length=512)
        return cache["tokenizer"].decode(tokens[0], skip_special_tokens=True)

    @staticmethod
    def _get_nllb_code(lang_key):
        codes = {
            "Kannada (ಕನ್ನಡ)": "kan_Knda",
            "Marathi (मराठी)": "mar_Deva",
            "Bengali (বাংলা)": "ben_Beng",
            "Malayalam (മലയാളം)": "mal_Mlym",
            "Tamil (தமிழ்)": "tam_Taml",
            "Konkani (कोंकणी)": "mar_Deva"  # Fallback to Marathi as NLLB lacks Konkani
        }
        return codes.get(lang_key, "hin_Deva")

    @staticmethod
    def sarvam_stt(wav_path, lang_code):
        api_key = os.environ.get("SARVAM_API_KEY", "")
        if not api_key:
            raise Exception("SARVAM_API_KEY not found in environment")
        with open(wav_path, "rb") as f:
            response = requests.post(AIService.SARVAM_STT_URL, 
                headers={"api-subscription-key": api_key},
                files={"file": (os.path.basename(wav_path), f, "audio/wav")},
                data={"model": "saarika:v2.5", "language_code": lang_code},
                timeout=30)
        return response.json().get("transcript", "").strip()

    @staticmethod
    def _load_audio_float32(wav_path):
        """Load a WAV file and return a mono float32 numpy array at 16 kHz."""
        import scipy.io.wavfile as wavfile
        sample_rate, audio = wavfile.read(wav_path)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16000:
            import scipy.signal as signal
            num_samples = int(len(audio) * 16000 / sample_rate)
            audio = signal.resample(audio, num_samples)
        return audio

    @staticmethod
    def detect_language(wav_path) -> str:
        """Use Whisper to detect the spoken language. Returns an ISO-639-1 code, e.g. 'hi', 'en', 'ta'."""
        wmodel = get_whisper("base")
        audio = AIService._load_audio_float32(wav_path)
        # Whisper.detect_language needs a padded/trimmed 30-second mel
        audio_padded = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio_padded).to(wmodel.device)
        _, probs = wmodel.detect_language(mel)
        detected = max(probs, key=probs.get)
        confidence = probs[detected]
        print(f"[Language detection] {detected} ({confidence:.2%})")
        return detected

    @staticmethod
    def whisper_stt(wav_path, lang_code=None):
        """Transcribe using Whisper with scipy audio loading — no ffmpeg required.
        lang_code=None means Whisper auto-detects the language."""
        wmodel = get_whisper("base")
        audio = AIService._load_audio_float32(wav_path)
        kwargs = {"fp16": False}
        if lang_code:
            kwargs["language"] = lang_code
        res = wmodel.transcribe(audio, **kwargs)
        return res["text"].strip()

    @staticmethod
    def is_language_consistent(text: str, lang_key: str) -> bool:
        """
        Best-effort check that transcript script roughly matches selected language.
        Helps catch obvious wrong-language STT outputs (e.g., Malayalam for Hindi).
        """
        if not text:
            return False
        if lang_key == "English":
            letters = [ch for ch in text if ch.isalpha()]
            if not letters:
                return True
            latin = sum(1 for ch in letters if ("a" <= ch.lower() <= "z"))
            return latin / len(letters) >= 0.6

        script_ranges = {
            # Devanagari family
            "Hindi (हिन्दी)": (0x0900, 0x097F),
            "Marathi (मराठी)": (0x0900, 0x097F),
            "Konkani (कोंकणी)": (0x0900, 0x097F),
            # Other Indic scripts
            "Bengali (বাংলা)": (0x0980, 0x09FF),
            "Tamil (தமிழ்)": (0x0B80, 0x0BFF),
            "Kannada (ಕನ್ನಡ)": (0x0C80, 0x0CFF),
            "Malayalam (മലയാളം)": (0x0D00, 0x0D7F),
        }
        if lang_key not in script_ranges:
            return True

        start, end = script_ranges[lang_key]
        letters = [ch for ch in text if ch.isalpha()]
        if not letters:
            return True
        in_script = sum(1 for ch in letters if start <= ord(ch) <= end)
        # Keep threshold modest for mixed-script / short transcripts.
        return (in_script / len(letters)) >= 0.35

    @staticmethod
    def generate_tts(text: str, lang_key: str) -> tuple[bytes, str]:
        """Generate TTS audio and return the audio bytes and content type."""
        if lang_key == "English":
            # Prefer gTTS for English, but fall back to Sarvam if gTTS fails.
            try:
                from gtts import gTTS
                from io import BytesIO
                tts = gTTS(text=text, lang="en")
                fp = BytesIO()
                tts.write_to_fp(fp)
                return fp.getvalue(), "audio/mpeg"
            except Exception as gtts_err:
                print(f"[generate_tts] gTTS failed for English, trying Sarvam fallback: {gtts_err}")
                api_key = os.environ.get("SARVAM_API_KEY", "")
                if not api_key:
                    raise
                response = requests.post(
                    AIService.SARVAM_TTS_URL,
                    headers={"api-subscription-key": api_key, "Content-Type": "application/json"},
                    json={"target_language_code": "en-IN", "text": text, "speaker": "meera"},
                    timeout=20
                )
                response.raise_for_status()
                import base64
                js = response.json()
                audio_b64 = js.get("audio_output", js.get("audios", [""])[0])
                audio_data = base64.b64decode(audio_b64)
                return audio_data, "audio/wav"

        sarvam_lang_codes = {
            "Hindi (हिन्दी)":   "hi-IN",
            "Kannada (ಕನ್ನಡ)": "kn-IN",
            "Marathi (मराठी)":  "mr-IN",
            "Bengali (বাংলা)":  "bn-IN",
            "Malayalam (മലയാളം)": "ml-IN",
            "Tamil (தமிழ்)": "ta-IN",
            "Konkani (कोंकणी)": "mr-IN", # fallback for Konkani
        }
        gtts_lang_codes = {
            "Hindi (हिन्दी)": "hi",
            "Kannada (ಕನ್ನಡ)": "kn",
            "Marathi (मराठी)": "mr",
            "Bengali (বাংলা)": "bn",
            "Malayalam (മലയാളം)": "ml",
            "Tamil (தமிழ்)": "ta",
            "Konkani (कोंकणी)": "mr",
        }

        # Prefer Sarvam for Indian language quality, then fall back to gTTS.
        api_key = os.environ.get("SARVAM_API_KEY", "")
        if api_key:
            try:
                lang_code = sarvam_lang_codes.get(lang_key, "hi-IN")
                response = requests.post(
                    AIService.SARVAM_TTS_URL,
                    headers={"api-subscription-key": api_key, "Content-Type": "application/json"},
                    json={"target_language_code": lang_code, "text": text, "speaker": "meera"},
                    timeout=20
                )
                response.raise_for_status()
                import base64
                js = response.json()
                audio_b64 = js.get("audio_output", js.get("audios", [""])[0])
                audio_data = base64.b64decode(audio_b64)
                return audio_data, "audio/wav"
            except Exception as sarvam_err:
                print(f"[generate_tts] Sarvam TTS failed for {lang_key}, falling back to gTTS: {sarvam_err}")

        from gtts import gTTS
        from io import BytesIO
        gtts_lang = gtts_lang_codes.get(lang_key, "hi")
        tts = gTTS(text=text, lang=gtts_lang)
        fp = BytesIO()
        tts.write_to_fp(fp)
        return fp.getvalue(), "audio/mpeg"

# ─────────────────────────────────────────────
#  Report Generator
# ─────────────────────────────────────────────

# Absolute path to the bundled Noto fonts directory
_FONTS_DIR = Path(__file__).parent / "fonts"

# Map of script-range start codepoints → font file
# We register one font per script family.
_SCRIPT_FONTS = [
    # (unicode_start, unicode_end, font_file, fpdf_family_name)
    (0x0900, 0x097F, "NotoSansDevanagari-Regular.ttf", "NotoDevanagari"),  # Devanagari (Hindi, Marathi, Konkani)
    (0x0980, 0x09FF, "NotoSansBengali-Regular.ttf",    "NotoBengali"),     # Bengali
    (0x0B80, 0x0BFF, "NotoSansTamil-Regular.ttf",      "NotoTamil"),       # Tamil
    (0x0C00, 0x0C7F, "NotoSansKannada-Regular.ttf",    "NotoKannada"),     # Kannada
    (0x0D00, 0x0D7F, "NotoSansMalayalam-Regular.ttf",  "NotoMalayalam"),   # Malayalam
]


def _build_unicode_pdf(title_line1: str, title_line2: str, transcripts: list) -> FPDF:
    """Build an FPDF document that correctly renders Indian-script text.

    Strategy: split each text field into segments by Unicode block, then
    render each segment with the appropriate Noto script font.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Register all script fonts once
    registered = set()
    for _, _, fname, family in _SCRIPT_FONTS:
        fpath = _FONTS_DIR / fname
        if family not in registered and fpath.exists():
            pdf.add_font(family, "", str(fpath))
            registered.add(family)

    # Also register the Latin Noto font (for headers and ASCII)
    latin_path = _FONTS_DIR / "NotoSans-Regular.ttf"
    bold_path   = _FONTS_DIR / "NotoSans-Bold.ttf"
    if latin_path.exists():
        pdf.add_font("NotoLatin", "",  str(latin_path))
    if bold_path.exists():
        pdf.add_font("NotoLatin", "B", str(bold_path))

    def use(style="", size=11):
        try:
            pdf.set_font("NotoLatin", style, size)
        except Exception:
            pdf.set_font("Helvetica", style, size)

    # Title
    use("B", 16)
    pdf.cell(0, 10, title_line1, new_x="LMARGIN", new_y="NEXT", align="C")
    use("", 10)
    pdf.cell(0, 8, title_line2, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)

    def _family_for_char(ch: str) -> str:
        cp = ord(ch)
        for start, end, _, family in _SCRIPT_FONTS:
            if start <= cp <= end:
                return family
        return "NotoLatin"

    def _render_text(text: str, font_size=10):
        """Render a string, switching fonts per Unicode block segment."""
        if not text:
            return
        # Split into segments of the same font family
        segments = []
        cur_family = _family_for_char(text[0])
        cur_chunk = text[0]
        for ch in text[1:]:
            fam = _family_for_char(ch)
            if fam == cur_family:
                cur_chunk += ch
            else:
                segments.append((cur_family, cur_chunk))
                cur_family, cur_chunk = fam, ch
        segments.append((cur_family, cur_chunk))

        # Write segments inline, wrapping manually with multi_cell for each chunk
        # For simplicity, concatenate segments into one write using the dominant font
        # (fpdf2 doesn't support mid-cell font switch easily without spans)
        # Best effort: use the family of the first non-Latin chunk, fall back to Latin
        dominant = next((f for f, _ in segments if f != "NotoLatin"), "NotoLatin")
        try:
            pdf.set_font(dominant, "", font_size)
        except Exception:
            pdf.set_font("NotoLatin", "", font_size)
        try:
            pdf.multi_cell(0, 6, text)
        except Exception as exc:
            print(f"[PDF render error, skipping line]: {exc}")

    for entry in transcripts:
        # Entry header
        use("B", 10)
        ts = entry.get("timestamp", "")
        role = entry.get("role", "").upper()
        pdf.cell(0, 6, f"[{ts}] {role}:", new_x="LMARGIN", new_y="NEXT")

        use("", 10)
        pdf.cell(15, 6, "Orig:", new_x="END", new_y="LAST")
        _render_text(entry.get("original", ""), font_size=10)

        use("", 10)
        pdf.cell(15, 6, "Tran:", new_x="END", new_y="LAST")
        _render_text(entry.get("translated", ""), font_size=10)

        pdf.ln(3)

    return pdf


class ReportGenerator:

    @staticmethod
    def generate_pdf(patient_id, patient_name, transcripts=None):
        """Generate a consultation PDF.

        If *transcripts* is provided (list of dicts), it is used directly —
        this lets the frontend pass the live in-memory session so the PDF
        always reflects what is on screen, not just what is saved to disk.
        """
        p_dir = PATIENTS_DIR / patient_id
        log_path = p_dir / "transcripts.json"
        pdf_path = p_dir / "consultation_history.pdf"

        if transcripts is None:
            if not log_path.exists():
                return None
            with open(log_path, "r", encoding="utf-8") as f:
                transcripts = json.load(f)

        title1 = f"Medical Consultation History - {patient_name}"
        title2 = f"Patient ID: {patient_id}  |  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        pdf = _build_unicode_pdf(title1, title2, transcripts)
        pdf.output(str(pdf_path))
        return pdf_path

    @staticmethod
    def archive_chat(patient_id, patient_name):
        p_dir = PATIENTS_DIR / patient_id
        log_path = p_dir / "transcripts.json"

        if not log_path.exists():
            return None

        with open(log_path, "r", encoding="utf-8") as f:
            transcripts = json.load(f)

        if not transcripts:
            return None

        history_dir = p_dir / "history"
        history_dir.mkdir(exist_ok=True)

        timestamp = int(datetime.now().timestamp())
        pdf_path = history_dir / f"chat_history_{timestamp}.pdf"

        title1 = f"Medical Consultation Archive - {patient_name}"
        title2 = f"Patient ID: {patient_id}  |  Archived: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        pdf = _build_unicode_pdf(title1, title2, transcripts)
        pdf.output(str(pdf_path))

        # Clear transcripts after archiving
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)

        return pdf_path

    @staticmethod
    def list_archives(patient_id):
        """Return a list of {filename, timestamp, size_kb} for all archives."""
        history_dir = PATIENTS_DIR / patient_id / "history"
        if not history_dir.exists():
            return []
        archives = []
        for p in sorted(history_dir.glob("chat_history_*.pdf"), reverse=True):
            ts_str = p.stem.replace("chat_history_", "")
            try:
                dt = datetime.fromtimestamp(int(ts_str)).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                dt = ts_str
            archives.append({
                "filename": p.name,
                "archived_at": dt,
                "size_kb": round(p.stat().st_size / 1024, 1)
            })
        return archives
