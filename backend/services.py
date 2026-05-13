import os
import json
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
import requests
import numpy as np
import scipy.io.wavfile as wav
from fpdf import FPDF
from pydub import AudioSegment
from medical_translation import translate_medical_native_pair, translate_medical_text
from backend.audio_pipeline import load_wav_float32

# ─────────────────────────────────────────────
#  Lazy Model Cache
# ─────────────────────────────────────────────
_MODEL_CACHE = {}

FASTER_WHISPER_MODEL = os.environ.get("FASTER_WHISPER_MODEL", "medium")
FASTER_WHISPER_DEVICE = os.environ.get("FASTER_WHISPER_DEVICE", "auto")
FASTER_WHISPER_COMPUTE_TYPE = os.environ.get("FASTER_WHISPER_COMPUTE_TYPE", "auto")
FASTER_WHISPER_CPU_THREADS = int(
    os.environ.get("FASTER_WHISPER_CPU_THREADS", str(max(1, min(8, os.cpu_count() or 1))))
)

def get_whisper(size: str | None = None):
    model_size = size or FASTER_WHISPER_MODEL
    key = f"faster_whisper_{model_size}_{FASTER_WHISPER_DEVICE}_{FASTER_WHISPER_COMPUTE_TYPE}"
    if key not in _MODEL_CACHE:
        from faster_whisper import WhisperModel
        _MODEL_CACHE[key] = WhisperModel(
            model_size,
            device=FASTER_WHISPER_DEVICE,
            compute_type=FASTER_WHISPER_COMPUTE_TYPE,
            cpu_threads=FASTER_WHISPER_CPU_THREADS,
        )
    return _MODEL_CACHE[key]


def _get_openai_whisper(size: str = "base"):
    key = f"openai_whisper_{size}"
    if key not in _MODEL_CACHE:
        import whisper
        _MODEL_CACHE[key] = whisper.load_model(size)
    return _MODEL_CACHE[key]

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

    def get_transcripts(self, patient_id: str) -> list[dict]:
        log_path = PATIENTS_DIR / patient_id / "transcripts.json"
        if not log_path.exists():
            return []
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

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

    _MEDICAL_STT_PROMPT = """Medical consultation transcript. Preserve exact symptoms, diagnoses, medicines, dosages, measurements, dates, durations, and body parts.
Common terms: fever, cough, cold, chest pain, breathing difficulty, wheezing, dizziness, nausea, vomiting, diarrhea, stomach pain, headache, weakness, swelling, allergy, infection, diabetes, hypertension, asthma, migraine, ECG, MRI, CT scan, X-ray, HbA1c, blood pressure, SpO2, paracetamol, ibuprofen, amoxicillin, azithromycin, metformin, insulin, amlodipine, telmisartan, atorvastatin, pantoprazole, omeprazole, salbutamol."""

    @staticmethod
    def build_conversation_context(patient_id: str | None, limit: int = 8) -> str:
        if not patient_id:
            return ""
        transcripts = PatientManager().get_transcripts(patient_id)[-limit:]
        lines = []
        for entry in transcripts:
            role = (entry.get("role") or "speaker").capitalize()
            original = (entry.get("original") or "").strip()
            translated = (entry.get("translated") or "").strip()
            if original:
                lines.append(f"{role}: {original}")
            if translated and translated != original:
                lines.append(f"{role} translated: {translated}")
        return "\n".join(lines)[-1800:]

    @staticmethod
    def build_medical_stt_prompt(patient_id: str | None = None, role: str | None = None) -> str:
        parts = [AIService._MEDICAL_STT_PROMPT]
        if role:
            parts.append(f"Current speaker role: {role}.")
        context = AIService.build_conversation_context(patient_id)
        if context:
            parts.append("Recent conversation memory:\n" + context)
        return "\n\n".join(parts)

    @staticmethod
    def correct_transcript(text: str, patient_id: str | None = None, role: str | None = None) -> str:
        """Send STT text through Sarvam's LLM for medical transcription correction."""
        api_key = os.environ.get("SARVAM_API_KEY", "")
        if not api_key:
            print("[correct_transcript] No SARVAM_API_KEY — skipping correction.")
            return text
        try:
            context = AIService.build_conversation_context(patient_id)
            user_content = f'"""\n{text}\n"""'
            if context:
                speaker = f" for the {role}" if role else ""
                user_content = (
                    f"Recent consultation memory{speaker}:\n{context}\n\n"
                    f"Correct this newest transcript only:\n{user_content}"
                )
            payload = {
                "model": "sarvam-m",
                "messages": [
                    {"role": "system", "content": AIService._CORRECTION_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content}
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
            return translate_medical_text(text, direction, lang_key)
        except Exception as e:
            return f"Error: {str(e)}"

    @staticmethod
    def translate_native_pair(text: str, source_lang_key: str, target_lang_key: str) -> str:
        try:
            return translate_medical_native_pair(text, source_lang_key, target_lang_key)
        except Exception as e:
            return f"Error: {str(e)}"

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
        _, audio = load_wav_float32(wav_path)
        return audio

    @staticmethod
    def detect_language_probs(wav_path) -> dict:
        """
        Faster-Whisper language-id confidence used for auto speaker detection.
        """
        audio = AIService._load_audio_float32(wav_path)
        try:
            wmodel = get_whisper()
            segments, info = wmodel.transcribe(
                audio,
                beam_size=1,
                language=None,
                task="transcribe",
                initial_prompt=AIService._MEDICAL_STT_PROMPT,
                vad_filter=False,
            )
            for _ in segments:
                break
            lang = getattr(info, "language", None) or "en"
            prob = float(getattr(info, "language_probability", 0.0) or 0.0)
            return {lang: prob}
        except Exception as faster_err:
            print(f"[Faster-Whisper language detection failed, using openai-whisper fallback]: {faster_err}")
            import whisper
            wmodel = _get_openai_whisper("base")
            audio_padded = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio_padded).to(wmodel.device)
            _, probs = wmodel.detect_language(mel)
            return dict(probs)

    @staticmethod
    def detect_language(wav_path) -> str:
        """Use Faster-Whisper to detect the spoken language. Returns an ISO-639-1 code."""
        probs = AIService.detect_language_probs(wav_path)
        detected = max(probs, key=probs.get)
        confidence = probs[detected]
        print(f"[Language detection] {detected} ({confidence:.2%})")
        return detected

    @staticmethod
    def whisper_stt(wav_path, lang_code=None, patient_id: str | None = None, role: str | None = None):
        """Transcribe using Whisper with scipy audio loading — no ffmpeg required.
        lang_code=None means Whisper auto-detects the language."""
        audio = AIService._load_audio_float32(wav_path)
        prompt = AIService.build_medical_stt_prompt(patient_id, role)
        try:
            wmodel = get_whisper()
            segments, _ = wmodel.transcribe(
                audio,
                language=lang_code,
                task="transcribe",
                beam_size=int(os.environ.get("FASTER_WHISPER_BEAM_SIZE", "5")),
                best_of=int(os.environ.get("FASTER_WHISPER_BEST_OF", "5")),
                temperature=float(os.environ.get("FASTER_WHISPER_TEMPERATURE", "0")),
                condition_on_previous_text=True,
                initial_prompt=prompt,
                vad_filter=False,
            )
            return " ".join(seg.text.strip() for seg in segments if seg.text).strip()
        except Exception as faster_err:
            print(f"[Faster-Whisper STT failed, using openai-whisper fallback]: {faster_err}")
            wmodel = _get_openai_whisper(os.environ.get("OPENAI_WHISPER_FALLBACK_MODEL", "base"))
            kwargs = {"fp16": False, "task": "transcribe", "initial_prompt": prompt}
            if lang_code:
                kwargs["language"] = lang_code
            res = wmodel.transcribe(audio, **kwargs)
            return res.get("text", "").strip()

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
