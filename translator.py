import tkinter as tk
from tkinter import ttk
import threading
import tempfile
import os
from datetime import datetime
import queue

# Load .env for SARVAM_API_KEY
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────
#  Lazy model cache  (loaded once per session)
# ─────────────────────────────────────────────
_model_cache: dict = {}

def _get_marian(model_name: str):
    from transformers import MarianMTModel, MarianTokenizer
    if model_name not in _model_cache:
        _model_cache[model_name] = {
            "tokenizer": MarianTokenizer.from_pretrained(model_name),
            "model":     MarianMTModel.from_pretrained(model_name),
        }
    return _model_cache[model_name]["tokenizer"], _model_cache[model_name]["model"]


def _get_nllb(model_name: str = "facebook/nllb-200-distilled-600M"):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    if model_name not in _model_cache:
        _model_cache[model_name] = {
            "tokenizer": AutoTokenizer.from_pretrained(model_name, use_fast=False),
            "model":     AutoModelForSeq2SeqLM.from_pretrained(model_name),
        }
    return _model_cache[model_name]["tokenizer"], _model_cache[model_name]["model"]


# ─────────────────────────────────────────────
#  Translation back-ends
# ─────────────────────────────────────────────
def translate_marian(text: str, model_name: str) -> str:
    import torch
    tokenizer, model = _get_marian(model_name)
    inputs = tokenizer([text], return_tensors="pt", padding=True,
                       truncation=True, max_length=512)
    with torch.no_grad():
        tokens = model.generate(**inputs, max_length=512,
                                num_beams=5, early_stopping=True)
    return tokenizer.decode(tokens[0], skip_special_tokens=True)


def _nllb_translate(text: str, src_lang: str, tgt_lang: str) -> str:
    import torch
    tokenizer, model = _get_nllb()
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=512)
    bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    with torch.no_grad():
        tokens = model.generate(**inputs, forced_bos_token_id=bos_id,
                                max_length=512, num_beams=4, early_stopping=True)
    return tokenizer.decode(tokens[0], skip_special_tokens=True)



# ─────────────────────────────────────────────
#  Speech-to-Text  (Whisper, offline — English only)
# ─────────────────────────────────────────────
_whisper_models: dict = {}  # cache keyed by model size

def _get_whisper(size: str = "base"):
    """Load and cache a Whisper model of the requested size."""
    if size not in _whisper_models:
        import whisper
        _whisper_models[size] = whisper.load_model(size)
    return _whisper_models[size]


# ─────────────────────────────────────────────
#  Speech-to-Text  (Sarvam AI — Indian languages)
#  API docs: https://docs.sarvam.ai/api-reference-docs/endpoints/speech-to-text
#  Set SARVAM_API_KEY in your .env file.
# ─────────────────────────────────────────────
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"

def sarvam_speech_to_text(wav_path: str, lang_code: str) -> str:
    """Transcribe audio using Sarvam AI (best for Indian languages)."""
    import requests

    api_key = os.environ.get("SARVAM_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "SARVAM_API_KEY not set.\n"
            "Add it to your .env file:\n"
            "  SARVAM_API_KEY=your_key_here"
        )

    with open(wav_path, "rb") as f:
        response = requests.post(
            SARVAM_STT_URL,
            headers={"api-subscription-key": api_key},
            files={"file": (os.path.basename(wav_path), f, "audio/wav")},
            data={"model": "saarika:v2.5", "language_code": lang_code},
            timeout=30,
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"Sarvam API error {response.status_code}: {response.text}"
        )

    data = response.json()
    transcript = data.get("transcript", "").strip()
    if not transcript:
        raise ValueError("Sarvam returned an empty transcript. Try speaking more clearly.")
    return transcript


# ─────────────────────────────────────────────
#  STT routing table
#    backend="sarvam"  → calls sarvam_speech_to_text()
#    backend="whisper" → calls local Whisper model
# ─────────────────────────────────────────────
_STT_CONFIG = {
    "Hindi (हिन्दी)":   {"backend": "sarvam", "lang": "hi-IN"},
    "Kannada (ಕನ್ನಡ)": {"backend": "sarvam", "lang": "kn-IN"},
    "Marathi (मराठी)":  {"backend": "sarvam", "lang": "mr-IN"},
    "Bengali (বাংলা)":  {"backend": "sarvam", "lang": "bn-IN"},
    "English":           {"backend": "whisper", "lang": "en", "model": "base"},
}


# ─────────────────────────────────────────────
#  Text-to-Speech  (gTTS, needs internet)
# ─────────────────────────────────────────────

# Map our language keys to BCP-47 codes for gTTS
_GTTS_LANG = {
    "Hindi (हिन्दी)":   "hi",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Marathi (मराठी)":  "mr",
    "Bengali (বাংলা)":  "bn",
    "English":           "en",
}


def speak_text(text: str, lang_key: str):
    """Convert text to speech and play it via gTTS."""
    from gtts import gTTS
    import pygame

    code = _GTTS_LANG.get(lang_key, "en")
    tts  = gTTS(text=text, lang=code, slow=False)

    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    tts.save(tmp.name)

    pygame.mixer.init()
    pygame.mixer.music.load(tmp.name)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)
    
    try:
        pygame.mixer.music.unload()
    except AttributeError:
        pass  # In case of older pygame version
    pygame.mixer.quit()
    
    try:
        os.unlink(tmp.name)
    except OSError:
        pass


#  Each entry has:
#    "native"     – display name with script
#    "en_to"      – fn to translate English → this language
#    "to_en"      – fn to translate this language → English
#    "font_native"– font for rendering native script
# ─────────────────────────────────────────────
LANGUAGES = {
    "Hindi (हिन्दी)": {
        "native":      "Hindi (हिन्दी)",
        "en_to":       lambda t: translate_marian(t, "Helsinki-NLP/opus-mt-en-hi"),
        "to_en":       lambda t: translate_marian(t, "Helsinki-NLP/opus-mt-hi-en"),
        "font_native": ("Nirmala UI", 14),
    },
    "Kannada (ಕನ್ನಡ)": {
        "native":      "Kannada (ಕನ್ನಡ)",
        "en_to":       lambda t: _nllb_translate(t, "eng_Latn", "kan_Knda"),
        "to_en":       lambda t: _nllb_translate(t, "kan_Knda", "eng_Latn"),
        "font_native": ("Noto Sans Kannada", 14),
    },
    "Marathi (मराठी)": {
        "native":      "Marathi (मराठी)",
        "en_to":       lambda t: _nllb_translate(t, "eng_Latn", "mar_Deva"),
        "to_en":       lambda t: _nllb_translate(t, "mar_Deva", "eng_Latn"),
        "font_native": ("Nirmala UI", 14),
    },
    "Bengali (বাংলা)": {
        "native":      "Bengali (বাংলা)",
        "en_to":       lambda t: _nllb_translate(t, "eng_Latn", "ben_Beng"),
        "to_en":       lambda t: _nllb_translate(t, "ben_Beng", "eng_Latn"),
        "font_native": ("Noto Sans Bengali", 14),
    },
}

MAX_CHARS  = 1000
HIST_LIMIT = 20

# ─────────────────────────────────────────────
#  Color / font constants
# ─────────────────────────────────────────────
BG      = "#1a1a2e"
CARD    = "#16213e"
ACCENT  = "#e94560"
TEXT    = "#eaeaea"
SUBTEXT = "#8899aa"
BTN_BG  = "#0f3460"
FONT_H  = ("Segoe UI", 13, "bold")
FONT_N  = ("Segoe UI", 11)
FONT_EN = ("Consolas", 11)


# ─────────────────────────────────────────────
#  App
# ─────────────────────────────────────────────
class TranslatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Offline Indian Language Translator")
        self.geometry("700x720")
        self.resizable(True, True)
        self.configure(bg=BG)
        self.history: list[dict] = []
        # direction: "en_to_native" or "native_to_en"
        self._direction = "en_to_native"
        self.is_recording = False
        self.audio_queue = None
        self.audio_stream = None
        self._build_ui()

    # ── UI construction ──────────────────────
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        outer = tk.Frame(self, bg=BG, padx=18, pady=14)
        outer.grid(sticky="nsew")
        outer.columnconfigure(0, weight=1)

        # Header
        hdr = tk.Frame(outer, bg=BG)
        hdr.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        tk.Label(hdr, text="🇮🇳  Offline Translator",
                 font=("Segoe UI", 18, "bold"), bg=BG, fg=ACCENT).pack(side="left")

        # ── Direction + Language row ──
        row1 = tk.Frame(outer, bg=BG)
        row1.grid(row=1, column=0, sticky="ew", pady=4)

        tk.Label(row1, text="Language:", font=FONT_N,
                 bg=BG, fg=TEXT).pack(side="left")

        self.lang_var = tk.StringVar(value=list(LANGUAGES.keys())[0])
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=CARD,
                        background=BTN_BG, foreground=TEXT, arrowcolor=ACCENT)
        cb = ttk.Combobox(row1, textvariable=self.lang_var,
                          values=list(LANGUAGES.keys()),
                          font=FONT_N, width=24, state="readonly")
        cb.pack(side="left", padx=8)
        cb.bind("<<ComboboxSelected>>", lambda _: self._update_direction_label())

        # Direction label + swap button
        self.dir_lbl = tk.Label(row1, text="",
                                font=("Segoe UI", 10), bg=BG, fg=SUBTEXT)
        self.dir_lbl.pack(side="left", padx=(4, 0))

        tk.Button(row1, text="⇄ Swap", font=("Segoe UI", 10),
                  bg=BTN_BG, fg=TEXT, relief="flat", padx=10, pady=3,
                  cursor="hand2", command=self._swap_direction).pack(side="left", padx=8)

        # ── Input card ──
        ic = tk.Frame(outer, bg=CARD)
        ic.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ic.columnconfigure(0, weight=1)

        # Input label row with 🎤 button
        ilbl_row = tk.Frame(ic, bg=CARD)
        ilbl_row.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 2))
        self.input_lbl = tk.Label(ilbl_row, text="Input", font=FONT_H,
                                  bg=CARD, fg=ACCENT)
        self.input_lbl.pack(side="left")
        self.mic_btn = tk.Button(ilbl_row, text="🎤 Speak", font=("Segoe UI", 9),
                                 bg=BTN_BG, fg=TEXT, relief="flat", padx=8, pady=2,
                                 cursor="hand2", command=self._start_mic)
        self.mic_btn.pack(side="right")
        self.input_box = tk.Text(
            ic, height=6, font=FONT_EN,
            bg="#0d1b2a", fg=TEXT, insertbackground=TEXT,
            relief="flat", padx=8, pady=6, wrap="word", undo=True)
        self.input_box.grid(row=1, column=0, sticky="ew", padx=10)
        self.input_box.bind("<KeyRelease>", self._on_input_change)

        ifoot = tk.Frame(ic, bg=CARD)
        ifoot.grid(row=2, column=0, sticky="ew", padx=10, pady=4)
        self.char_lbl = tk.Label(ifoot, text=f"0 / {MAX_CHARS}",
                                 font=("Segoe UI", 9), bg=CARD, fg=SUBTEXT)
        self.char_lbl.pack(side="right")
        tk.Button(ifoot, text="✕ Clear", font=("Segoe UI", 9),
                  bg=CARD, fg=SUBTEXT, bd=0, cursor="hand2",
                  command=self._clear_input).pack(side="left")

        # ── Translate button row ──
        btn_row = tk.Frame(outer, bg=BG)
        btn_row.grid(row=3, column=0, pady=10)
        self.translate_btn = tk.Button(
            btn_row, text="Translate  ➜", font=("Segoe UI", 12, "bold"),
            bg=ACCENT, fg="white", relief="flat", padx=20, pady=7,
            cursor="hand2", command=self._start_translation)
        self.translate_btn.pack(side="left", padx=6)

        tk.Button(btn_row, text="📋 Copy Result", font=("Segoe UI", 10),
                  bg=BTN_BG, fg=TEXT, relief="flat", padx=12, pady=7,
                  cursor="hand2", command=self._copy_result).pack(side="left", padx=6)

        tk.Button(btn_row, text="🕑 History", font=("Segoe UI", 10),
                  bg=BTN_BG, fg=TEXT, relief="flat", padx=12, pady=7,
                  cursor="hand2", command=self._show_history).pack(side="left", padx=6)

        # ── Status bar ──
        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(outer, textvariable=self.status_var, font=("Segoe UI", 9),
                 bg=BG, fg=SUBTEXT, anchor="w").grid(row=4, column=0,
                                                      sticky="ew", pady=(0, 4))

        # ── Output card ──
        oc = tk.Frame(outer, bg=CARD)
        oc.grid(row=5, column=0, sticky="nsew", pady=(0, 8))
        oc.columnconfigure(0, weight=1)
        oc.columnconfigure(1, weight=0)
        outer.rowconfigure(5, weight=1)
        self.output_lbl = tk.Label(oc, text="Translation", font=FONT_H,
                                   bg=CARD, fg=ACCENT)
        self.output_lbl.grid(row=0, column=0, sticky="w", padx=10, pady=(8, 2))

        # 🔊 button on output card header
        self.speak_btn = tk.Button(oc, text="🔊 Speak", font=("Segoe UI", 9),
                                   bg=BTN_BG, fg=TEXT, relief="flat", padx=8, pady=2,
                                   cursor="hand2", command=self._start_speak)
        self.speak_btn.grid(row=0, column=1, sticky="e", padx=10, pady=(8, 2))
        self.output_box = tk.Text(
            oc, height=6, font=("Nirmala UI", 14),
            bg="#0d1b2a", fg=TEXT, relief="flat", padx=8, pady=6,
            wrap="word", state="disabled")
        self.output_box.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 10))

        # Now that all widgets exist, set initial labels
        self._update_direction_label()

    # ── direction helpers ─────────────────────
    def _lang_short(self) -> str:
        """Return language name without script e.g. 'Kannada'."""
        return self.lang_var.get().split("(")[0].strip()

    def _update_direction_label(self):
        lang = self._lang_short()
        if self._direction == "en_to_native":
            self.dir_lbl.config(text=f"English  →  {lang}")
            self.input_lbl.config(text="English Input")
            self.output_lbl.config(text=f"{lang} Translation")
            self.input_box.config(font=FONT_EN)
        else:
            self.dir_lbl.config(text=f"{lang}  →  English")
            self.input_lbl.config(text=f"{lang} Input")
            self.output_lbl.config(text="English Translation")
            cfg = LANGUAGES[self.lang_var.get()]
            self.input_box.config(font=cfg["font_native"])

    def _swap_direction(self):
        # Move current output text to input
        current_output = self.output_box.get("1.0", tk.END).strip()

        self._direction = (
            "native_to_en" if self._direction == "en_to_native" else "en_to_native"
        )
        self._update_direction_label()

        # Populate input with previous result if available
        if current_output and not current_output.startswith("❌"):
            self.input_box.delete("1.0", tk.END)
            self.input_box.insert(tk.END, current_output)
            self._on_input_change()

        # Clear output
        self._set_output("", font=None)
        self.status_var.set("Direction swapped.")

    # ── helpers ──────────────────────────────
    def _on_input_change(self, _event=None):
        n = len(self.input_box.get("1.0", tk.END).strip())
        self.char_lbl.config(
            text=f"{n} / {MAX_CHARS}",
            fg=ACCENT if n > MAX_CHARS else SUBTEXT)

    def _clear_input(self):
        self.input_box.delete("1.0", tk.END)
        self._on_input_change()

    def _set_output(self, text: str, font):
        self.output_box.config(state="normal")
        self.output_box.delete("1.0", tk.END)
        if font:
            self.output_box.config(font=font)
        self.output_box.insert(tk.END, text)
        self.output_box.config(state="disabled")

    def _copy_result(self):
        result = self.output_box.get("1.0", tk.END).strip()
        if result:
            self.clipboard_clear()
            self.clipboard_append(result)
            self.status_var.set("✔ Copied to clipboard.")

    # ── translation (runs in background thread) ──
    def _start_translation(self):
        src = self.input_box.get("1.0", tk.END).strip()
        if not src:
            self.status_var.set("⚠  Please enter some text first.")
            return
        if len(src) > MAX_CHARS:
            self.status_var.set(f"⚠  Text exceeds {MAX_CHARS} character limit.")
            return

        self.translate_btn.config(state="disabled", text="Translating…")
        self.status_var.set("⏳  Loading model and translating…")

        lang_key  = self.lang_var.get()
        direction = self._direction

        def worker():
            try:
                cfg = LANGUAGES[lang_key]
                fn  = cfg["en_to"] if direction == "en_to_native" else cfg["to_en"]
                out_font = cfg["font_native"] if direction == "en_to_native" \
                           else FONT_EN
                result = fn(src)
                self.after(0, self._on_success, result, out_font, src, lang_key, direction)
            except Exception as exc:
                self.after(0, self._on_error, str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_success(self, result: str, font, src: str, lang_key: str, direction: str):
        self._set_output(result, font)
        lang = lang_key.split("(")[0].strip()
        arrow = f"English → {lang}" if direction == "en_to_native" else f"{lang} → English"
        self.status_var.set(f"✔  {arrow} done.")
        self.translate_btn.config(state="normal", text="Translate  ➜")
        self.history.insert(0, {
            "ts":    datetime.now().strftime("%H:%M:%S"),
            "arrow": arrow,
            "src":   src[:80] + ("…" if len(src) > 80 else ""),
            "tgt":   result,
        })
        if len(self.history) > HIST_LIMIT:
            self.history.pop()

    # ── 🎤 microphone input ───────────────────
    def _audio_callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status)
        if self.audio_queue:
            self.audio_queue.put(indata.copy())

    def _start_mic(self):
        """Toggle recording audio and transcribe into the input box."""
        if not self.is_recording:
            import sounddevice as sd
            self.audio_queue = queue.Queue()
            self.is_recording = True
            self.mic_btn.config(text="🛑 Stop Rec", fg="red")
            self.status_var.set("🎤  Recording... Click 'Stop Rec' when done.")
            
            self.audio_stream = sd.InputStream(
                samplerate=16000, channels=1, dtype="int16", callback=self._audio_callback
            )
            self.audio_stream.start()
        else:
            self.is_recording = False
            self.audio_stream.stop()
            self.audio_stream.close()
            self.mic_btn.config(state="disabled", text="⏳ Transcribing...", fg=TEXT)
            self.status_var.set("🎤  Transcribing audio...")

            if self._direction == "en_to_native":
                stt_cfg = _STT_CONFIG.get("English")
            else:
                stt_cfg = _STT_CONFIG.get(
                    self.lang_var.get(), _STT_CONFIG["English"]
                )

            def worker():
                try:
                    import numpy as np
                    import scipy.io.wavfile as wav
                    import tempfile
                    import os

                    audio_data = []
                    while not self.audio_queue.empty():
                        audio_data.append(self.audio_queue.get())

                    if not audio_data:
                        self.after(0, self._on_mic_error, "No audio recorded.")
                        return

                    audio_np = np.concatenate(audio_data, axis=0)

                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmp.close()
                    wav.write(tmp.name, 16000, audio_np)

                    backend   = stt_cfg["backend"]
                    lang_code = stt_cfg["lang"]

                    if backend == "sarvam":
                        self.after(0, lambda: self.status_var.set(
                            "🎤  Sending to Sarvam AI — please wait..."
                        ))
                        text = sarvam_speech_to_text(tmp.name, lang_code)
                    else:
                        model_size = stt_cfg.get("model", "base")
                        self.after(0, lambda s=model_size: self.status_var.set(
                            f"🎤  Loading Whisper '{s}' model — please wait..."
                        ))
                        wmodel = _get_whisper(model_size)
                        result = wmodel.transcribe(
                            tmp.name,
                            language=lang_code,
                            task="transcribe",
                            fp16=False,
                        )
                        text = result["text"]
                    
                    try:
                        os.unlink(tmp.name)
                    except OSError:
                        pass
                        
                    text = text.strip()
                    self.after(0, self._on_mic_done, text)
                except Exception as exc:
                    self.after(0, self._on_mic_error, str(exc))

            threading.Thread(target=worker, daemon=True).start()

    def _on_mic_done(self, text: str):
        self.input_box.delete("1.0", tk.END)
        self.input_box.insert(tk.END, text)
        self._on_input_change()
        self.mic_btn.config(state="normal", text="🎤 Speak", fg=TEXT)
        self.status_var.set("🎤  Transcription done. Auto-translating...")
        self._start_translation()

    def _on_mic_error(self, msg: str):
        self.mic_btn.config(state="normal", text="🎤 Speak", fg=TEXT)
        self.status_var.set(f"❌  Mic error: {msg}")

    # ── 🔊 speak output ───────────────────────
    def _start_speak(self):
        """Speak the translated output via gTTS."""
        result = self.output_box.get("1.0", tk.END).strip()
        if not result or result.startswith("❌"):
            self.status_var.set("⚠  Nothing to speak yet.")
            return

        # Determine language of the output
        if self._direction == "en_to_native":
            lang_key = self.lang_var.get()
        else:
            lang_key = "English"

        self.speak_btn.config(state="disabled", text="🔊 Speaking…")
        self.status_var.set("🔊  Speaking…")

        def worker():
            try:
                speak_text(result, lang_key)
                self.after(0, lambda: self.speak_btn.config(
                    state="normal", text="🔊 Speak"))
                self.after(0, lambda: self.status_var.set("🔊  Done."))
            except Exception as exc:
                self.after(0, self._on_speak_error, str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_speak_error(self, msg: str):
        self.speak_btn.config(state="normal", text="🔊 Speak")
        self.status_var.set(f"❌  TTS error: {msg}")

    def _on_error(self, msg: str):
        self._set_output(f"❌ Error: {msg}", font=FONT_EN)
        self.status_var.set("❌  Translation failed.")
        self.translate_btn.config(state="normal", text="Translate  ➜")

    # ── history window ────────────────────────
    def _show_history(self):
        win = tk.Toplevel(self)
        win.title("Translation History")
        win.geometry("600x400")
        win.configure(bg=BG)

        tk.Label(win, text="Recent Translations", font=FONT_H,
                 bg=BG, fg=ACCENT).pack(pady=10)

        if not self.history:
            tk.Label(win, text="No history yet.", font=FONT_N,
                     bg=BG, fg=SUBTEXT).pack()
            return

        frame = tk.Frame(win, bg=BG)
        frame.pack(fill="both", expand=True, padx=14, pady=4)
        sb = tk.Scrollbar(frame)
        sb.pack(side="right", fill="y")
        lb = tk.Text(frame, font=("Consolas", 10), bg=CARD, fg=TEXT,
                     relief="flat", padx=8, pady=6, wrap="word",
                     yscrollcommand=sb.set)
        lb.pack(fill="both", expand=True)
        sb.config(command=lb.yview)

        for item in self.history:
            lb.insert(tk.END,
                      f"[{item['ts']}]  {item['arrow']}\n"
                      f"  IN:  {item['src']}\n"
                      f"  OUT: {item['tgt']}\n\n")
        lb.config(state="disabled")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = TranslatorApp()
    app.mainloop()