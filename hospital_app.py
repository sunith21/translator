import os
import webbrowser
import urllib.parse
import json
import threading
import tempfile
import queue
import sqlite3
import shutil
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
from dotenv import load_dotenv
from fpdf import FPDF
from pydub import AudioSegment
import requests
import whisper
import torch
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from transformers import MarianMTModel, MarianTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
from gtts import gTTS
import pygame

# Load environment variables
load_dotenv()

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
#  Constants & Configuration
# ─────────────────────────────────────────────
ACCENT = "#e94560"
BG_COLOR = "#1a1a2e"
CARD_COLOR = "#16213e"
TEXT_COLOR = "#eaeaea"
SUBTEXT_COLOR = "#8899aa"

PATIENTS_DIR = Path("patients")
PATIENTS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
#  Database & File Management
# ─────────────────────────────────────────────
class PatientManager:
    def __init__(self):
        self.db_path = "hospital_data.db"
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    phone TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_visit TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Migrate existing DBs that lack the phone column
            try:
                conn.execute("ALTER TABLE patients ADD COLUMN phone TEXT DEFAULT ''")
            except Exception:
                pass  # Column already exists

    def add_patient(self, patient_id, name, phone=""):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO patients (id, name, phone, last_visit) VALUES (?, ?, ?, ?)",
                (patient_id, name, phone, datetime.now()),
            )
        
        # Create directory structure
        p_dir = PATIENTS_DIR / patient_id
        p_dir.mkdir(exist_ok=True)
        (p_dir / "recordings").mkdir(exist_ok=True)
        
        # Init transcript log if not exists
        log_path = p_dir / "transcripts.json"
        if not log_path.exists():
            with open(log_path, "w") as f:
                json.dump([], f)

    def get_patient(self, patient_id):
        with sqlite3.connect(self.db_path) as conn:
            res = conn.execute(
                "SELECT id, name, phone, last_visit FROM patients WHERE id = ?",
                (patient_id,),
            ).fetchone()
            if res:
                return {"id": res[0], "name": res[1], "phone": res[2] or "", "last_visit": res[3]}
        return None

    def search_patients(self, query):
        with sqlite3.connect(self.db_path) as conn:
            res = conn.execute("SELECT id, name FROM patients WHERE id LIKE ? OR name LIKE ? LIMIT 10",
                               (f"%{query}%", f"%{query}%")).fetchall()
            return [{"id": r[0], "name": r[1]} for r in res]

    def get_recent_patients(self):
        with sqlite3.connect(self.db_path) as conn:
            res = conn.execute("SELECT id, name, last_visit FROM patients ORDER BY last_visit DESC LIMIT 5").fetchall()
            return [{"id": r[0], "name": r[1], "last_visit": r[2]} for r in res]

# ─────────────────────────────────────────────
#  PDF Report Generator
# ─────────────────────────────────────────────
class ReportGenerator:
    # DejaVuSans supports Unicode (Devanagari, Kannada, Bengali, etc.)
    # Download it once into the app directory if not present.
    FONT_PATH = Path("DejaVuSans.ttf")
    FONT_BOLD_PATH = Path("DejaVuSans-Bold.ttf")

    @staticmethod
    def _ensure_fonts():
        """Download DejaVuSans TTF if not already present."""
        urls = {
            ReportGenerator.FONT_PATH: "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf",
            ReportGenerator.FONT_BOLD_PATH: "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans-Bold.ttf",
        }
        for path, url in urls.items():
            if not path.exists():
                try:
                    import urllib.request
                    urllib.request.urlretrieve(url, str(path))
                except Exception:
                    pass  # Will fall back to ASCII-only Latin font

    @staticmethod
    def update_pdf(patient_id, patient_name):
        p_dir = PATIENTS_DIR / patient_id
        log_path = p_dir / "transcripts.json"
        pdf_path = p_dir / "consultation_history.pdf"

        if not log_path.exists():
            return

        with open(log_path, "r", encoding="utf-8") as f:
            transcripts = json.load(f)

        ReportGenerator._ensure_fonts()
        has_unicode_font = ReportGenerator.FONT_PATH.exists()

        pdf = FPDF()
        pdf.add_page()

        if has_unicode_font:
            pdf.add_font("DejaVu", "", str(ReportGenerator.FONT_PATH))
            pdf.add_font("DejaVu", "B", str(ReportGenerator.FONT_BOLD_PATH))
            head_font = ("DejaVu", "B", 16)
            sub_font  = ("DejaVu", "", 10)
            bold_font = ("DejaVu", "B", 12)
            entry_bold = ("DejaVu", "B", 10)
            entry_reg  = ("DejaVu", "",  10)
        else:
            head_font = ("Arial", "B", 16)
            sub_font  = ("Arial", "", 10)
            bold_font = ("Arial", "B", 12)
            entry_bold = ("Arial", "B", 10)
            entry_reg  = ("Arial", "",  10)

        pdf.set_font(*head_font)
        pdf.cell(0, 10, f"Medical Consultation History - {patient_name}",
                 new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_font(*sub_font)
        pdf.cell(0, 8, f"Patient ID: {patient_id}",
                 new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(6)

        current_date = None
        for entry in transcripts:
            entry_date = entry["timestamp"].split(" ")[0]
            if entry_date != current_date:
                current_date = entry_date
                pdf.set_font(*bold_font)
                pdf.set_fill_color(220, 230, 245)
                pdf.cell(0, 9, f"  Date: {current_date}",
                         new_x="LMARGIN", new_y="NEXT", fill=True)
                pdf.ln(2)

            pdf.set_font(*entry_bold)
            pdf.cell(0, 5,
                     f"[{entry['timestamp'].split(' ')[1]}]  {entry['role'].upper()}:",
                     new_x="LMARGIN", new_y="NEXT")
            pdf.set_font(*entry_reg)
            pdf.multi_cell(0, 5, f"  Original:   {entry['original']}")
            pdf.multi_cell(0, 5, f"  Translated: {entry['translated']}")
            pdf.ln(3)

        pdf.output(str(pdf_path))

# ─────────────────────────────────────────────
#  AI Services Integration
# ─────────────────────────────────────────────
class AIService:
    SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
    SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

    @staticmethod
    def translate(text, direction, lang_key):
        # direction: "en_to_native" or "native_to_en"
        # For simplicity, using same logic as translator.py
        try:
            if direction == "en_to_native":
                if lang_key == "Hindi (हिन्दी)":
                    res = AIService._translate_marian(text, "Helsinki-NLP/opus-mt-en-hi")
                else:
                    # Default to NLLB for others
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
            "Bengali (বাংলা)": "ben_Beng"
        }
        return codes.get(lang_key, "hin_Deva")

    @staticmethod
    def sarvam_stt(wav_path, lang_code):
        api_key = os.environ.get("SARVAM_API_KEY", "")
        with open(wav_path, "rb") as f:
            response = requests.post(AIService.SARVAM_STT_URL, 
                headers={"api-subscription-key": api_key},
                files={"file": (os.path.basename(wav_path), f, "audio/wav")},
                data={"model": "saarika:v2.5", "language_code": lang_code},
                timeout=30)
        return response.json().get("transcript", "").strip()

    @staticmethod
    def speak(text, lang_key, is_indian=True):
        """
        Generate speech, play it, and return saved audio path.
        """
        if is_indian:
            return AIService.sarvam_tts(text, lang_key)
        return AIService.gtts_speak(text)

    @staticmethod
    def sarvam_tts(text, lang_key, output_path=None):
        api_key = os.environ.get("SARVAM_API_KEY", "")
        lang_map = {"Hindi (हिन्दी)": "hi-IN", "Kannada (ಕನ್ನಡ)": "kn-IN", "Marathi (मराठी)": "mr-IN", "Bengali (বাংলা)": "bn-IN"}
        lang_code = lang_map.get(lang_key, "hi-IN")
        
        response = requests.post(AIService.SARVAM_TTS_URL,
            headers={"api-subscription-key": api_key, "Content-Type": "application/json"},
            json={"target_language_code": lang_code, "text": text, "speaker": "meera"})
        
        if response.status_code == 200:
            import base64
            audio_data = base64.b64decode(response.json().get("audio_output", ""))
            if output_path:
                with open(output_path, "wb") as f:
                    f.write(audio_data)
                AIService._play_audio(output_path)
                return output_path
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(audio_data)
            tmp.close()
            AIService._play_audio(tmp.name)
            os.unlink(tmp.name)
            return None
        return None

    @staticmethod
    def gtts_speak(text, output_path=None):
        tts = gTTS(text=text, lang="en")
        if output_path:
            tts.save(output_path)
            AIService._play_audio(output_path)
            return output_path
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()
        tts.save(tmp.name)
        AIService._play_audio(tmp.name)
        os.unlink(tmp.name)
        return None

    @staticmethod
    def _play_audio(path):
        pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
        pygame.mixer.quit()

# ─────────────────────────────────────────────
#  Main Application UI
# ─────────────────────────────────────────────
class HospitalApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("Dark")
        self.title("AI Medical Consultation System")
        self.geometry("1000x700")
        
        self.db = PatientManager()
        self.current_patient = None
        self.is_recording = False
        self.audio_queue = None
        self.audio_stream = None
        self.session_recordings = [] # To store np arrays of current session
        
        # Grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main_content()
        
        self.show_dashboard()

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(self.sidebar, text="🏥 CLINIC AI", font=("Segoe UI", 20, "bold"), text_color=ACCENT).pack(pady=20)
        
        self.btn_dash = ctk.CTkButton(self.sidebar, text="Dashboard", command=self.show_dashboard, fg_color="transparent", text_color=TEXT_COLOR, anchor="w")
        self.btn_dash.pack(fill="x", padx=10, pady=5)
        
        self.btn_new = ctk.CTkButton(self.sidebar, text="+ New Consultation", command=self.show_new_patient, fg_color="transparent", text_color=TEXT_COLOR, anchor="w")
        self.btn_new.pack(fill="x", padx=10, pady=5)
        
        self.btn_search = ctk.CTkButton(self.sidebar, text="Search Patient", command=self.show_search, fg_color="transparent", text_color=TEXT_COLOR, anchor="w")
        self.btn_search.pack(fill="x", padx=10, pady=5)

    def _build_main_content(self):
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)

    def clear_main(self):
        for widget in self.main_container.winfo_children():
            widget.destroy()

    # ─── Pages ───

    def show_dashboard(self):
        self.clear_main()
        dash = ctk.CTkFrame(self.main_container, fg_color="transparent")
        dash.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(dash, text="Clinic Dashboard", font=("Segoe UI", 24, "bold")).pack(anchor="w", pady=(0, 20))
        
        # Recent Patients
        recents = ctk.CTkFrame(dash, fg_color=CARD_COLOR, corner_radius=10)
        recents.pack(fill="x", pady=10)
        ctk.CTkLabel(recents, text="Recent Patients", font=("Segoe UI", 16, "bold"), text_color=ACCENT).pack(anchor="w", padx=15, pady=10)
        
        patients = self.db.get_recent_patients()
        if not patients:
            ctk.CTkLabel(recents, text="No recent records found.", text_color=SUBTEXT_COLOR).pack(pady=10)
        else:
            for p in patients:
                p_row = ctk.CTkFrame(recents, fg_color="transparent")
                p_row.pack(fill="x", padx=15, pady=2)
                ctk.CTkLabel(p_row, text=f"{p['name']} (ID: {p['id']})", font=("Segoe UI", 12)).pack(side="left")
                ctk.CTkButton(p_row, text="Open", width=60, height=24, command=lambda pid=p['id']: self.load_patient(pid)).pack(side="right")

    def show_new_patient(self):
        self.clear_main()
        form = ctk.CTkFrame(self.main_container, fg_color=CARD_COLOR, corner_radius=10, width=400)
        form.place(relx=0.5, rely=0.4, anchor="center")

        ctk.CTkLabel(form, text="New Consultation", font=("Segoe UI", 18, "bold"), text_color=ACCENT).pack(pady=20)

        self.entry_name = ctk.CTkEntry(form, placeholder_text="Patient Name", width=300)
        self.entry_name.pack(pady=10)

        self.entry_id = ctk.CTkEntry(form, placeholder_text="Unique Patient ID", width=300)
        self.entry_id.pack(pady=10)

        self.entry_phone = ctk.CTkEntry(form, placeholder_text="WhatsApp Number (e.g. 919876543210)", width=300)
        self.entry_phone.pack(pady=10)

        ctk.CTkLabel(
            form,
            text="Include country code, no '+' or spaces",
            text_color=SUBTEXT_COLOR,
            font=("Segoe UI", 10),
        ).pack()

        ctk.CTkButton(form, text="Start Session", command=self.handle_new_session, fg_color=ACCENT).pack(pady=20)

    def show_search(self):
        self.clear_main()
        search_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        search_frame.grid(row=0, column=0, sticky="nsew")
        search_frame.grid_columnconfigure(0, weight=1)
        search_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(search_frame, text="Patient Search", font=("Segoe UI", 24, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 10))

        self.search_entry = ctk.CTkEntry(search_frame, placeholder_text="Enter Name or ID...", width=400)
        self.search_entry.grid(row=1, column=0, sticky="w", pady=(0, 10))
        self.search_entry.bind("<KeyRelease>", self.update_search_results)

        self.results_box = ctk.CTkScrollableFrame(search_frame, fg_color=CARD_COLOR)
        self.results_box.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        search_frame.grid_rowconfigure(2, weight=1)

    def update_search_results(self, event=None):
        q = self.search_entry.get()
        for widget in self.results_box.winfo_children():
            widget.destroy()
        
        if not q: return
        
        results = self.db.search_patients(q)
        for r in results:
            btn = ctk.CTkButton(self.results_box, text=f"{r['name']} - {r['id']}", 
                                fg_color="transparent", text_color=TEXT_COLOR, anchor="w",
                                command=lambda pid=r['id']: self.load_patient(pid))
            btn.pack(fill="x", pady=2)

    def handle_new_session(self):
        name = self.entry_name.get().strip()
        pid = self.entry_id.get().strip()
        phone = self.entry_phone.get().strip()
        if name and pid:
            self.db.add_patient(pid, name, phone)
            self.load_patient(pid)

    def load_patient(self, pid):
        p_data = self.db.get_patient(pid)
        if p_data:
            # Update last_visit so this patient rises to top of Recent Patients
            with sqlite3.connect(self.db.db_path) as conn:
                conn.execute("UPDATE patients SET last_visit = ? WHERE id = ?",
                             (datetime.now(), pid))
            self.current_patient = p_data
            self.show_consultation_view()

    def show_consultation_view(self):
        self.clear_main()
        self.session_recordings = []  # Reset for new session
        view = ctk.CTkFrame(self.main_container, fg_color="transparent")
        view.grid(row=0, column=0, sticky="nsew")
        
        # Header
        hdr = ctk.CTkFrame(view, fg_color=CARD_COLOR, height=60)
        hdr.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(hdr, text=f"Patient: {self.current_patient['name']}", font=("Segoe UI", 16, "bold")).pack(side="left", padx=20, pady=10)
        ctk.CTkLabel(hdr, text=f"ID: {self.current_patient['id']}", text_color=SUBTEXT_COLOR).pack(side="left", padx=10, pady=10)
        
        # Main area: Split into Chat and History
        split = ctk.CTkFrame(view, fg_color="transparent")
        split.pack(fill="both", expand=True)
        
        # Left: Live Consultation
        live = ctk.CTkFrame(split, fg_color=CARD_COLOR, corner_radius=10)
        live.pack(side="left", fill="both", expand=True, padx=(0, 10))
        self.lang_var = ctk.StringVar(value="Hindi (हिन्दी)")
        self.lang_cb = ctk.CTkComboBox(live, values=["Hindi (हिन्दी)", "Kannada (ಕನ್ನಡ)", "Marathi (मराठी)", "Bengali (বাংলা)"], variable=self.lang_var)
        self.lang_cb.pack(pady=5)

        self.chat_box = ctk.CTkTextbox(live, fg_color="#0d1b2a", font=("Segoe UI", 12))
        self.chat_box.pack(fill="both", expand=True, padx=10, pady=5)
        
        ctrl = ctk.CTkFrame(live, fg_color="transparent")
        ctrl.pack(fill="x", pady=10)
        self.btn_doc = ctk.CTkButton(ctrl, text="🎤 Doctor Speak", fg_color="#0f3460", command=lambda: self.start_stt_flow("doctor"))
        self.btn_doc.pack(side="left", expand=True, padx=5)
        self.btn_pat = ctk.CTkButton(ctrl, text="🎤 Patient Speak", fg_color="#0f3460", command=lambda: self.start_stt_flow("patient"))
        self.btn_pat.pack(side="left", expand=True, padx=5)
        
        # Right: History / PDF
        hist = ctk.CTkFrame(split, width=300, fg_color=CARD_COLOR, corner_radius=10)
        hist.pack(side="right", fill="y")
        ctk.CTkLabel(hist, text="Session Info", font=("Segoe UI", 14, "bold")).pack(pady=10)
        ctk.CTkButton(hist, text="Open Patient PDF", fg_color="transparent", border_width=1, command=self.open_pdf).pack(pady=5, padx=20, fill="x")
        ctk.CTkButton(
            hist,
            text="📱 Send via WhatsApp",
            fg_color="#25D366",
            hover_color="#1ebe57",
            text_color="#ffffff",
            command=self.send_to_whatsapp,
        ).pack(pady=5, padx=20, fill="x")
        ctk.CTkButton(hist, text="View Past Recordings", fg_color="transparent", border_width=1, command=self._view_recordings).pack(pady=5, padx=20, fill="x")
        ctk.CTkButton(hist, text="End Session", fg_color="#611", command=self.end_session).pack(side="bottom", pady=20, padx=20, fill="x")

    # ─── Consultation Logic ───

    def start_stt_flow(self, role):
        if self.is_recording:
            self.stop_recording(role)
        else:
            self.start_recording(role)

    def start_recording(self, role):
        self.is_recording = True
        self.audio_queue = queue.Queue()
        if role == "doctor":
            self.btn_doc.configure(text="🛑 Stop Doctor", fg_color="red")
            self.btn_pat.configure(state="disabled")
        else:
            self.btn_pat.configure(text="🛑 Stop Patient", fg_color="red")
            self.btn_doc.configure(state="disabled")

        def callback(indata, frames, time, status):
            self.audio_queue.put(indata.copy())

        self.audio_stream = sd.InputStream(samplerate=16000, channels=1, dtype="int16", callback=callback)
        self.audio_stream.start()

    def stop_recording(self, role):
        self.is_recording = False
        self.audio_stream.stop()
        self.audio_stream.close()
        self.btn_doc.configure(text="🎤 Doctor Speak", fg_color="#0f3460", state="normal")
        self.btn_pat.configure(text="🎤 Patient Speak", fg_color="#0f3460", state="normal")

        # Process audio
        data = []
        while not self.audio_queue.empty():
            data.append(self.audio_queue.get())
        
        if not data: return
        
        audio_np = np.concatenate(data, axis=0)
        self.session_recordings.append(audio_np) # Save for session mp3

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        wav.write(tmp.name, 16000, audio_np)
        
        threading.Thread(target=self.process_audio, args=(tmp.name, role), daemon=True).start()

    def process_audio(self, wav_path, role):
        lang_key = self.lang_var.get()
        lang_codes = {"Hindi (हिन्दी)": "hi-IN", "Kannada (ಕನ್ನಡ)": "kn-IN", "Marathi (मराठी)": "mr-IN", "Bengali (বাংলা)": "bn-IN"}
        
        try:
            # 1. STT
            if role == "doctor":
                # Doctor speaks English
                wmodel = get_whisper("base")
                res = wmodel.transcribe(wav_path, language="en", fp16=False)
                original = res["text"].strip()
                direction = "en_to_native"
            else:
                # Patient speaks native
                original = AIService.sarvam_stt(wav_path, lang_codes.get(lang_key, "hi-IN"))
                direction = "native_to_en"

            # 2. Translate
            translated = AIService.translate(original, direction, lang_key)

            # 3. Save per-utterance audio copy
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            rec_dir = PATIENTS_DIR / self.current_patient["id"] / "recordings"
            rec_dir.mkdir(exist_ok=True)
            utt_path = rec_dir / f"utt_{ts}_{role}.wav"
            try:
                shutil.copy(wav_path, utt_path)
                audio_path_str = str(utt_path)
            except Exception:
                audio_path_str = None

            # 4. Save & Display (including audio path if available)
            self.after(0, self.update_transcript, role, original, translated, audio_path_str)

            # 5. Speak and save translated TTS playback for both sides
            tts_path = rec_dir / f"tts_{ts}_{role}.wav"
            if role == "doctor":
                saved_tts = AIService.sarvam_tts(translated, lang_key, output_path=str(tts_path))
            else:
                saved_tts = AIService.gtts_speak(translated, output_path=str(tts_path.with_suffix(".mp3")))

            if saved_tts:
                self.after(0, self._attach_translated_audio_to_last_entry, str(saved_tts))

        except Exception as e:
            self.after(0, lambda msg=str(e): self.chat_box.insert("end", f"\nError: {msg}\n"))
        finally:
            os.unlink(wav_path)

    def update_transcript(self, role, original, translated, audio_path=None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {"role": role, "original": original, "translated": translated, "timestamp": timestamp}
        if audio_path is not None:
            entry["audio_path"] = audio_path
        
        # Save to JSON
        p_dir = PATIENTS_DIR / self.current_patient["id"]
        log_path = p_dir / "transcripts.json"
        with open(log_path, "r+") as f:
            data = json.load(f)
            data.append(entry)
            f.seek(0)
            json.dump(data, f)
            f.truncate()

        # Display
        self.chat_box.insert("end", f"[{timestamp}] {role.upper()}\nIn: {original}\nOut: {translated}\n\n")
        self.chat_box.see("end")

    def _attach_translated_audio_to_last_entry(self, translated_audio_path):
        """
        Store translated speech recording path on the latest transcript item.
        """
        p_dir = PATIENTS_DIR / self.current_patient["id"]
        log_path = p_dir / "transcripts.json"
        try:
            with open(log_path, "r+") as f:
                data = json.load(f)
                if data:
                    data[-1]["translated_audio_path"] = translated_audio_path
                    f.seek(0)
                    json.dump(data, f)
                    f.truncate()
        except Exception:
            pass

    def open_pdf(self):
        ReportGenerator.update_pdf(self.current_patient["id"], self.current_patient["name"])
        os.startfile(PATIENTS_DIR / self.current_patient["id"] / "consultation_history.pdf")

    def send_to_whatsapp(self):
        """Fully automatic: drives WhatsApp Web via Selenium to attach and
        send the patient PDF report without any manual steps."""
        pid   = self.current_patient["id"]
        name  = self.current_patient["name"]
        phone = self.current_patient.get("phone", "").strip()

        if not phone:
            messagebox.showerror(
                "No Phone Number",
                "No WhatsApp number is saved for this patient.\n\n"
                "Please create a new consultation and enter the patient's\n"
                "WhatsApp number (country code + number, no '+').",
            )
            return

        # Regenerate / refresh the PDF first
        ReportGenerator.update_pdf(pid, name)
        pdf_path = str((PATIENTS_DIR / pid / "consultation_history.pdf").absolute())

        # ── Progress window ───────────────────────────────────────────────
        prog = ctk.CTkToplevel(self)
        prog.title("Sending via WhatsApp")
        prog.geometry("420x180")
        prog.resizable(False, False)
        prog.grab_set()

        ctk.CTkLabel(prog, text="📱 Sending Report via WhatsApp",
                     font=("Segoe UI", 15, "bold")).pack(pady=(20, 6))
        status_lbl = ctk.CTkLabel(prog, text="Initialising…",
                                  font=("Segoe UI", 12), text_color=SUBTEXT_COLOR,
                                  wraplength=390)
        status_lbl.pack(pady=4, padx=20)
        progress_bar = ctk.CTkProgressBar(prog, mode="indeterminate", width=370)
        progress_bar.pack(pady=10)
        progress_bar.start()

        def set_status(msg):
            self.after(0, lambda: status_lbl.configure(text=msg))

        def close_prog():
            self.after(0, lambda: (progress_bar.stop(), prog.destroy()))

        # ── Selenium worker (runs in background thread) ───────────────────
        def _selenium_worker():
            driver = None
            try:
                from selenium import webdriver
                from selenium.webdriver.common.by import By
                from selenium.webdriver.support.ui import WebDriverWait
                from selenium.webdriver.support import expected_conditions as EC
                from selenium.webdriver.chrome.options import Options
                from selenium.webdriver.chrome.service import Service
                from webdriver_manager.chrome import ChromeDriverManager
                import time

                phone_clean = phone.lstrip("+").replace(" ", "").replace("-", "")

                # Persistent Chrome profile so QR code is only scanned ONCE
                session_dir = str(Path("whatsapp_chrome_session").absolute())
                Path(session_dir).mkdir(exist_ok=True)

                options = Options()
                options.add_argument(f"--user-data-dir={session_dir}")
                options.add_argument("--profile-directory=Default")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                # Keep browser window visible so user can scan QR on first run
                options.add_experimental_option("excludeSwitches", ["enable-automation"])
                options.add_experimental_option("useAutomationExtension", False)

                set_status("Starting Chrome — please wait…")
                service = Service(ChromeDriverManager().install())
                driver  = webdriver.Chrome(service=service, options=options)

                # Build companion message
                caption = (
                    f"\U0001f3e5 *Medical Consultation Report*\n"
                    f"Patient : {name}\n"
                    f"ID      : {pid}\n"
                    f"Date    : {datetime.now().strftime('%d %b %Y')}"
                )

                set_status("Opening WhatsApp Web…\n(Scan QR code if prompted — only needed once)")
                driver.get(f"https://web.whatsapp.com/send?phone={phone_clean}&text={urllib.parse.quote(caption)}")

                wait = WebDriverWait(driver, 120)   # up to 2 min for QR scan

                # ── Wait until the chat message-box is visible ─────────────
                set_status("Waiting for chat to open… (scan QR if shown)")
                MSGBOX_XPATHS = [
                    '//div[@contenteditable="true"][@data-tab="10"]',
                    '//div[@contenteditable="true"][@title="Type a message"]',
                    '//div[@contenteditable="true"][contains(@class,"copyable-text")]',
                ]
                msg_box = None
                for xp in MSGBOX_XPATHS:
                    try:
                        msg_box = wait.until(EC.presence_of_element_located((By.XPATH, xp)))
                        break
                    except Exception:
                        pass
                if msg_box is None:
                    raise RuntimeError("Could not find WhatsApp chat input. Is the number valid?")

                time.sleep(2)   # let the page settle

                # ── Click the Attach (paperclip) button ────────────────────
                set_status("Attaching PDF…")
                ATTACH_SELECTORS = [
                    (By.CSS_SELECTOR, 'div[title="Attach"]'),
                    (By.CSS_SELECTOR, 'button[title="Attach"]'),
                    (By.XPATH, '//*[@data-testid="clip"]'),
                    (By.XPATH, '//*[@data-icon="attach-menu-plus"]'),
                    (By.XPATH, '//span[@data-icon="clip"]'),
                ]
                attach_btn = None
                for by, sel in ATTACH_SELECTORS:
                    try:
                        attach_btn = WebDriverWait(driver, 6).until(
                            EC.element_to_be_clickable((by, sel)))
                        break
                    except Exception:
                        pass
                if attach_btn is None:
                    raise RuntimeError("Could not find the Attach button in WhatsApp Web.")

                attach_btn.click()
                time.sleep(1.5)

                # ── Feed the PDF path directly into the hidden file <input> ─
                FILE_INPUT_SELECTORS = [
                    'input[type="file"][accept="*"]',
                    'input[type="file"]',
                ]
                file_input = None
                for sel in FILE_INPUT_SELECTORS:
                    try:
                        file_input = WebDriverWait(driver, 6).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, sel)))
                        break
                    except Exception:
                        pass
                if file_input is None:
                    raise RuntimeError("Could not find file input in WhatsApp Web.")

                file_input.send_keys(pdf_path)
                time.sleep(3)   # wait for preview to render

                # ── Click the final Send button ────────────────────────────
                set_status("Sending report…")
                SEND_SELECTORS = [
                    (By.CSS_SELECTOR,  'span[data-icon="send"]'),
                    (By.XPATH,         '//*[@data-testid="send"]'),
                    (By.XPATH,         '//span[@data-icon="send"]'),
                    (By.CSS_SELECTOR,  '[data-testid="send"]'),
                ]
                send_btn = None
                for by, sel in SEND_SELECTORS:
                    try:
                        send_btn = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((by, sel)))
                        break
                    except Exception:
                        pass
                if send_btn is None:
                    raise RuntimeError("Could not find Send button after file was attached.")

                send_btn.click()
                time.sleep(4)   # wait for upload to complete

                set_status("✅ Report sent successfully!")
                time.sleep(2)
                driver.quit()
                close_prog()
                self.after(0, lambda: messagebox.showinfo(
                    "WhatsApp", f"Report sent to {phone_clean} successfully!"))

            except Exception as exc:
                if driver:
                    try:
                        driver.quit()
                    except Exception:
                        pass
                close_prog()
                self.after(0, lambda e=str(exc): messagebox.showerror(
                    "WhatsApp Error",
                    f"Could not send report automatically:\n\n{e}\n\n"
                    "Tips:\n"
                    "• Make sure Chrome is installed.\n"
                    "• Check that the phone number includes the country code.\n"
                    "• If it's your first run, you must scan the WhatsApp QR code once.",
                ))

        threading.Thread(target=_selenium_worker, daemon=True).start()

    def end_session(self):
        # Save session recording
        if self.session_recordings:
            full_audio = np.concatenate(self.session_recordings, axis=0)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            rec_dir = PATIENTS_DIR / self.current_patient["id"] / "recordings"
            wav_path = rec_dir / f"session_{ts}.wav"
            wav.write(str(wav_path), 16000, full_audio)

            # Try MP3 export (requires ffmpeg); fall back to keeping .wav
            try:
                audio = AudioSegment.from_wav(str(wav_path))
                mp3_path = wav_path.with_suffix(".mp3")
                audio.export(str(mp3_path), format="mp3")
                os.unlink(str(wav_path))   # Remove .wav only if MP3 succeeded
            except Exception:
                # ffmpeg not installed — keep the .wav file, rename for clarity
                wav_path.rename(rec_dir / f"session_{ts}_audio.wav")

        ReportGenerator.update_pdf(self.current_patient["id"], self.current_patient["name"])
        self.session_recordings = []
        self.show_dashboard()

    def _view_recordings(self):
        rec_dir = PATIENTS_DIR / self.current_patient["id"] / "recordings"
        if not rec_dir.exists(): return
        
        # Include both full-session and per-utterance recordings (wav/mp3)
        files = list(rec_dir.glob("*.mp3")) + list(rec_dir.glob("*.wav"))
        if not files:
            messagebox.showinfo("Recordings", "No recordings found for this patient.")
            return

        win = ctk.CTkToplevel(self)
        win.title("Past Recordings")
        win.geometry("400x300")
        
        ctk.CTkLabel(win, text="Session & Utterance Recordings", font=("Segoe UI", 16, "bold")).pack(pady=10)
        
        scroll = ctk.CTkScrollableFrame(win)
        scroll.pack(fill="both", expand=True, padx=10, pady=10)
        
        for f in sorted(files, reverse=True):
            name = f.name
            # Simple labeling: distinguish full sessions vs single utterances
            if name.startswith("session_"):
                label_text = f"Full session - {name}"
            elif name.startswith("utt_"):
                # Expected format: utt_YYYYmmdd_HHMMSS_role.ext
                parts = name.split("_")
                if len(parts) >= 3:
                    ts_part = parts[1]
                    role_part = parts[2].split(".")[0]  # role with extension removed
                    label_text = f"Utterance ({role_part}) at {ts_part}"
                else:
                    label_text = f"Utterance - {name}"
            else:
                label_text = name

            f_frame = ctk.CTkFrame(scroll, fg_color="transparent")
            f_frame.pack(fill="x", pady=2)
            ctk.CTkLabel(f_frame, text=label_text, font=("Segoe UI", 11)).pack(side="left")
            ctk.CTkButton(f_frame, text="▶ Play", width=50, height=24, command=lambda path=str(f): threading.Thread(target=AIService._play_audio, args=(path,), daemon=True).start()).pack(side="right")
            ctk.CTkButton(f_frame, text="📂 Open", width=50, height=24, command=lambda path=str(f): os.startfile(os.path.dirname(path))).pack(side="right", padx=5)

if __name__ == "__main__":
    app = HospitalApp()
    app.mainloop()
