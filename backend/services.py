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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_visit TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def add_patient(self, patient_id, name):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO patients (id, name, last_visit) VALUES (?, ?, ?)",
                         (patient_id, name, datetime.now()))
        
        p_dir = PATIENTS_DIR / patient_id
        p_dir.mkdir(exist_ok=True)
        (p_dir / "recordings").mkdir(exist_ok=True)
        
        log_path = p_dir / "transcripts.json"
        if not log_path.exists():
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def get_patient(self, patient_id):
        with sqlite3.connect(self.db_path) as conn:
            res = conn.execute("SELECT id, name, last_visit FROM patients WHERE id = ?", (patient_id,)).fetchone()
            if res:
                return {"id": res[0], "name": res[1], "last_visit": res[2]}
        return None

    def search_patients(self, query):
        with sqlite3.connect(self.db_path) as conn:
            res = conn.execute("SELECT id, name FROM patients WHERE id LIKE ? OR name LIKE ? LIMIT 10",
                               (f"%{query}%", f"%{query}%")).fetchall()
            return [{"id": r[0], "name": r[1]} for r in res]

    def get_recent_patients(self):
        with sqlite3.connect(self.db_path) as conn:
            res = conn.execute("SELECT id, name, last_visit FROM patients ORDER BY last_visit DESC LIMIT 10").fetchall()
            return [{"id": r[0], "name": r[1], "last_visit": r[2]} for r in res]

    def update_last_visit(self, patient_id):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE patients SET last_visit = ? WHERE id = ?", (datetime.now(), patient_id))

# ─────────────────────────────────────────────
#  AI Service
# ─────────────────────────────────────────────
class AIService:
    SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
    SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

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
    def whisper_stt(wav_path, lang_code=None):
        """Transcribe using Whisper with scipy audio loading — no ffmpeg required.
        lang_code=None means Whisper auto-detects the language."""
        import numpy as np
        import scipy.io.wavfile as wavfile

        wmodel = get_whisper("base")

        # Load WAV with scipy (avoids ffmpeg dependency)
        sample_rate, audio = wavfile.read(wav_path)

        # Convert to float32 in [-1, 1]
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype == np.float32:
            pass  # already correct
        else:
            audio = audio.astype(np.float32)

        # Make mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample to 16000 Hz if needed
        if sample_rate != 16000:
            import scipy.signal as signal
            num_samples = int(len(audio) * 16000 / sample_rate)
            audio = signal.resample(audio, num_samples)

        kwargs = {"fp16": False}
        if lang_code:
            kwargs["language"] = lang_code

        res = wmodel.transcribe(audio, **kwargs)
        return res["text"].strip()

# ─────────────────────────────────────────────
#  Report Generator
# ─────────────────────────────────────────────
class ReportGenerator:
    @staticmethod
    def generate_pdf(patient_id, patient_name):
        p_dir = PATIENTS_DIR / patient_id
        log_path = p_dir / "transcripts.json"
        pdf_path = p_dir / "consultation_history.pdf"

        if not log_path.exists():
            return None

        with open(log_path, "r", encoding="utf-8") as f:
            transcripts = json.load(f)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, f"Medical Consultation History - {patient_name}", ln=True, align="C")
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 8, f"Patient ID: {patient_id}", ln=True, align="C")
        pdf.ln(6)

        for entry in transcripts:
            pdf.set_font("Arial", "B", 10)
            pdf.cell(0, 5, f"[{entry['timestamp']}] {entry['role'].upper()}:", ln=True)
            pdf.set_font("Arial", "", 10)
            
            original_text = f"Original: {entry['original']}"
            translated_text = f"Translated: {entry['translated']}"
            
            try:
                pdf.multi_cell(0, 5, original_text)
            except Exception:
                try:
                    pdf.multi_cell(0, 5, original_text.encode('latin-1', 'replace').decode('latin-1'))
                except Exception:
                    pdf.multi_cell(0, 5, "[Original Text contains unsupported characters]")
                
            try:
                pdf.multi_cell(0, 5, translated_text)
            except Exception:
                try:
                    pdf.multi_cell(0, 5, translated_text.encode('latin-1', 'replace').decode('latin-1'))
                except Exception:
                    pdf.multi_cell(0, 5, "[Translated Text contains unsupported characters]")
                
            pdf.ln(3)

        pdf.output(str(pdf_path))
        return pdf_path

    @staticmethod
    def archive_chat(patient_id, patient_name):
        from datetime import datetime
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
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, f"Medical Consultation Archive - {patient_name}", ln=True, align="C")
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 8, f"Patient ID: {patient_id} | Archived: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
        pdf.ln(6)

        for entry in transcripts:
            pdf.set_font("Arial", "B", 10)
            pdf.cell(0, 5, f"[{entry['timestamp']}] {entry['role'].upper()}:", ln=True)
            pdf.set_font("Arial", "", 10)
            
            # FPDF standard fonts do not support complex Unicode (like Hindi/Kannada)
            # If it throws an error, fallback to replacing unencodable chars with '?'
            original_text = f"Original: {entry['original']}"
            translated_text = f"Translated: {entry['translated']}"
            
            try:
                pdf.multi_cell(0, 5, original_text)
            except Exception:
                try:
                    pdf.multi_cell(0, 5, original_text.encode('latin-1', 'replace').decode('latin-1'))
                except Exception:
                    pdf.multi_cell(0, 5, "[Original Text contains unsupported characters]")
                
            try:
                pdf.multi_cell(0, 5, translated_text)
            except Exception:
                try:
                    pdf.multi_cell(0, 5, translated_text.encode('latin-1', 'replace').decode('latin-1'))
                except Exception:
                    pdf.multi_cell(0, 5, "[Translated Text contains unsupported characters]")
                
            pdf.ln(3)

        pdf.output(str(pdf_path))
        
        # Clear transcripts
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
            
        return pdf_path
