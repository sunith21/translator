import re

with open('translator.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add import
if 'import customtkinter as ctk' not in content:
    content = content.replace('import tkinter as tk', 'import tkinter as tk\nimport customtkinter as ctk')

# Update constants
content = re.sub(
    r'BG\s*=\s*"#1a1a2e"\s*\nCARD\s*=\s*"#16213e"\s*\nACCENT\s*=\s*"#e94560"\s*\nTEXT\s*=\s*"#eaeaea"\s*\nSUBTEXT\s*=\s*"#8899aa"\s*\nBTN_BG\s*=\s*"#0f3460"\s*\nFONT_H\s*=\s*\("Segoe UI", 13, "bold"\)\s*\nFONT_N\s*=\s*\("Segoe UI", 11\)\s*\nFONT_EN\s*=\s*\("Consolas", 11\)',
    'ACCENT  = "#e94560"\nFONT_H  = ("Segoe UI", 14, "bold")\nFONT_N  = ("Segoe UI", 12)\nFONT_EN = ("Consolas", 12)',
    content
)

# New App Class
new_app_class = """class TranslatorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        
        self.title("Offline Indian Language Translator")
        self.geometry("700x740")
        self.resizable(True, True)
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

        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.grid(sticky="nsew", padx=18, pady=14)
        outer.columnconfigure(0, weight=1)

        # Header
        hdr = ctk.CTkFrame(outer, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        ctk.CTkLabel(hdr, text="🇮🇳 Offline Translator",
                 font=("Segoe UI", 20, "bold"), text_color=ACCENT).pack(side="left")

        # ── Direction + Language row ──
        row1 = ctk.CTkFrame(outer, fg_color="transparent")
        row1.grid(row=1, column=0, sticky="ew", pady=4)

        ctk.CTkLabel(row1, text="Language:", font=FONT_N).pack(side="left")

        self.lang_var = ctk.StringVar(value=list(LANGUAGES.keys())[0])
        cb = ctk.CTkComboBox(row1, variable=self.lang_var,
                          values=list(LANGUAGES.keys()),
                          font=FONT_N, width=200, state="readonly", command=self._on_combo_selected)
        cb.pack(side="left", padx=8)

        # Direction label + swap button
        self.dir_lbl = ctk.CTkLabel(row1, text="",
                                font=("Segoe UI", 12), text_color="gray")
        self.dir_lbl.pack(side="left", padx=(4, 0))

        ctk.CTkButton(row1, text="⇄ Swap", font=("Segoe UI", 12),
                  width=70, fg_color="#0f3460", hover_color="#1a5299",
                  command=self._swap_direction).pack(side="left", padx=8)

        # ── Input card ──
        ic = ctk.CTkFrame(outer, corner_radius=10)
        ic.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ic.columnconfigure(0, weight=1)

        # Input label row with 🎤 button
        ilbl_row = ctk.CTkFrame(ic, fg_color="transparent")
        ilbl_row.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 2))
        self.input_lbl = ctk.CTkLabel(ilbl_row, text="Input", font=FONT_H, text_color=ACCENT)
        self.input_lbl.pack(side="left")
        self.mic_btn = ctk.CTkButton(ilbl_row, text="🎤 Speak", font=("Segoe UI", 12),
                                 width=80, fg_color="#0f3460", hover_color="#1a5299",
                                 command=self._start_mic)
        self.mic_btn.pack(side="right")
        self.input_box = ctk.CTkTextbox(
            ic, height=120, font=FONT_EN, wrap="word")
        self.input_box.grid(row=1, column=0, sticky="ew", padx=10)
        self.input_box.bind("<KeyRelease>", self._on_input_change)

        ifoot = ctk.CTkFrame(ic, fg_color="transparent")
        ifoot.grid(row=2, column=0, sticky="ew", padx=10, pady=4)
        self.char_lbl = ctk.CTkLabel(ifoot, text=f"0 / {MAX_CHARS}",
                                 font=("Segoe UI", 11), text_color="gray")
        self.char_lbl.pack(side="right")
        ctk.CTkButton(ifoot, text="✕ Clear", font=("Segoe UI", 12),
                  width=60, fg_color="transparent", text_color="gray", hover_color="#2a2d3e",
                  command=self._clear_input).pack(side="left")

        # ── Translate button row ──
        btn_row = ctk.CTkFrame(outer, fg_color="transparent")
        btn_row.grid(row=3, column=0, pady=15)
        self.translate_btn = ctk.CTkButton(
            btn_row, text="Translate  ➜", font=("Segoe UI", 14, "bold"),
            fg_color=ACCENT, hover_color="#ff5e7e", height=35,
            command=self._start_translation)
        self.translate_btn.pack(side="left", padx=6)

        ctk.CTkButton(btn_row, text="📋 Copy Result", font=("Segoe UI", 12),
                  width=110, fg_color="#0f3460", hover_color="#1a5299", height=35,
                  command=self._copy_result).pack(side="left", padx=6)

        ctk.CTkButton(btn_row, text="🕑 History", font=("Segoe UI", 12),
                  width=90, fg_color="#0f3460", hover_color="#1a5299", height=35,
                  command=self._show_history).pack(side="left", padx=6)

        # ── Status bar ──
        self.status_var = ctk.StringVar(value="Ready.")
        ctk.CTkLabel(outer, textvariable=self.status_var, font=("Segoe UI", 12),
                 text_color="gray", anchor="w").grid(row=4, column=0,
                                                      sticky="ew", pady=(0, 4))

        # ── Output card ──
        oc = ctk.CTkFrame(outer, corner_radius=10)
        oc.grid(row=5, column=0, sticky="nsew", pady=(0, 8))
        oc.columnconfigure(0, weight=1)
        oc.columnconfigure(1, weight=0)
        outer.rowconfigure(5, weight=1)
        self.output_lbl = ctk.CTkLabel(oc, text="Translation", font=FONT_H, text_color=ACCENT)
        self.output_lbl.grid(row=0, column=0, sticky="w", padx=10, pady=(8, 2))

        # 🔊 button on output card header
        self.speak_btn = ctk.CTkButton(oc, text="🔊 Speak", font=("Segoe UI", 12),
                                   width=80, fg_color="#0f3460", hover_color="#1a5299",
                                   command=self._start_speak)
        self.speak_btn.grid(row=0, column=1, sticky="e", padx=10, pady=(8, 2))
        self.output_box = ctk.CTkTextbox(
            oc, height=120, font=("Nirmala UI", 16), wrap="word", state="disabled")
        self.output_box.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 10))

        # Now that all widgets exist, set initial labels
        self._update_direction_label()

    def _on_combo_selected(self, choice):
        self._update_direction_label()

    # ── direction helpers ─────────────────────
    def _lang_short(self) -> str:
        return self.lang_var.get().split("(")[0].strip()

    def _update_direction_label(self):
        lang = self._lang_short()
        if self._direction == "en_to_native":
            self.dir_lbl.configure(text=f"English  →  {lang}")
            self.input_lbl.configure(text="English Input")
            self.output_lbl.configure(text=f"{lang} Translation")
            self.input_box.configure(font=FONT_EN)
        else:
            self.dir_lbl.configure(text=f"{lang}  →  English")
            self.input_lbl.configure(text=f"{lang} Input")
            self.output_lbl.configure(text="English Translation")
            cfg = LANGUAGES[self.lang_var.get()]
            self.input_box.configure(font=cfg["font_native"])

    def _swap_direction(self):
        current_output = self.output_box.get("1.0", tk.END).strip()
        self._direction = "native_to_en" if self._direction == "en_to_native" else "en_to_native"
        self._update_direction_label()

        if current_output and not current_output.startswith("❌"):
            self.input_box.delete("1.0", tk.END)
            self.input_box.insert(tk.END, current_output)
            self._on_input_change()

        self._set_output("", font=None)
        self.status_var.set("Direction swapped.")

    # ── helpers ──────────────────────────────
    def _on_input_change(self, _event=None):
        n = len(self.input_box.get("1.0", tk.END).strip())
        self.char_lbl.configure(
            text=f"{n} / {MAX_CHARS}",
            text_color=ACCENT if n > MAX_CHARS else "gray")

    def _clear_input(self):
        self.input_box.delete("1.0", tk.END)
        self._on_input_change()

    def _set_output(self, text: str, font):
        self.output_box.configure(state="normal")
        self.output_box.delete("1.0", tk.END)
        if font:
            self.output_box.configure(font=font)
        self.output_box.insert(tk.END, text)
        self.output_box.configure(state="disabled")

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

        self.translate_btn.configure(state="disabled", text="Translating…")
        self.status_var.set("⏳  Loading model and translating…")

        lang_key  = self.lang_var.get()
        direction = self._direction

        def worker():
            try:
                cfg = LANGUAGES[lang_key]
                fn  = cfg["en_to"] if direction == "en_to_native" else cfg["to_en"]
                out_font = cfg["font_native"] if direction == "en_to_native" else FONT_EN
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
        self.translate_btn.configure(state="normal", text="Translate  ➜")
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
        if status:
            print(status)
        if self.audio_queue:
            self.audio_queue.put(indata.copy())

    def _start_mic(self):
        if not self.is_recording:
            import sounddevice as sd
            self.audio_queue = queue.Queue()
            self.is_recording = True
            self.mic_btn.configure(text="🛑 Stop Rec", fg_color="red", hover_color="#cc0000")
            self.status_var.set("🎤  Recording... Click 'Stop Rec' when done.")
            
            self.audio_stream = sd.InputStream(
                samplerate=16000, channels=1, dtype="int16", callback=self._audio_callback
            )
            self.audio_stream.start()
        else:
            self.is_recording = False
            self.audio_stream.stop()
            self.audio_stream.close()
            self.mic_btn.configure(state="disabled", text="⏳ Transcribing...", fg_color="#0f3460")
            self.status_var.set("🎤  Transcribing audio...")

            stt_cfg = _STT_CONFIG.get("English") if self._direction == "en_to_native" else _STT_CONFIG.get(self.lang_var.get(), _STT_CONFIG["English"])

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
                        self.after(0, lambda: self.status_var.set("🎤  Sending to Sarvam AI — please wait..."))
                        text = sarvam_speech_to_text(tmp.name, lang_code)
                    else:
                        model_size = stt_cfg.get("model", "base")
                        self.after(0, lambda s=model_size: self.status_var.set(f"🎤  Loading Whisper '{s}' model — please wait..."))
                        wmodel = _get_whisper(model_size)
                        result = wmodel.transcribe(tmp.name, language=lang_code, task="transcribe", fp16=False)
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
        self.mic_btn.configure(state="normal", text="🎤 Speak", fg_color="#0f3460", hover_color="#1a5299")
        self.status_var.set("🎤  Transcription done. Auto-translating...")
        self._start_translation()

    def _on_mic_error(self, msg: str):
        self.mic_btn.configure(state="normal", text="🎤 Speak", fg_color="#0f3460", hover_color="#1a5299")
        self.status_var.set(f"❌  Mic error: {msg}")

    # ── 🔊 speak output ───────────────────────
    def _start_speak(self):
        result = self.output_box.get("1.0", tk.END).strip()
        if not result or result.startswith("❌"):
            self.status_var.set("⚠  Nothing to speak yet.")
            return

        lang_key = self.lang_var.get() if self._direction == "en_to_native" else "English"

        self.speak_btn.configure(state="disabled", text="🔊 Speaking…")
        self.status_var.set("🔊  Speaking…")

        def worker():
            try:
                speak_text(result, lang_key)
                self.after(0, lambda: self.speak_btn.configure(state="normal", text="🔊 Speak"))
                self.after(0, lambda: self.status_var.set("🔊  Done."))
            except Exception as exc:
                self.after(0, self._on_speak_error, str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_speak_error(self, msg: str):
        self.speak_btn.configure(state="normal", text="🔊 Speak")
        self.status_var.set(f"❌  TTS error: {msg}")

    def _on_error(self, msg: str):
        self._set_output(f"❌ Error: {msg}", font=FONT_EN)
        self.status_var.set("❌  Translation failed.")
        self.translate_btn.configure(state="normal", text="Translate  ➜")

    # ── history window ────────────────────────
    def _show_history(self):
        win = ctk.CTkToplevel(self)
        win.title("Translation History")
        win.geometry("600x400")
        
        # Center or adjust window
        win.transient(self)

        ctk.CTkLabel(win, text="Recent Translations", font=FONT_H, text_color=ACCENT).pack(pady=10)

        if not self.history:
            ctk.CTkLabel(win, text="No history yet.", font=FONT_N, text_color="gray").pack()
            return

        frame = ctk.CTkFrame(win, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=14, pady=4)
        
        lb = ctk.CTkTextbox(frame, font=("Consolas", 12), wrap="word")
        lb.pack(fill="both", expand=True)

        for item in self.history:
            lb.insert(tk.END,
                      f"[{item['ts']}]  {item['arrow']}\\n"
                      f"  IN:  {item['src']}\\n"
                      f"  OUT: {item['tgt']}\\n\\n")
        lb.configure(state="disabled")"""

# Find the start of class TranslatorApp
start_idx = content.find('class TranslatorApp(tk.Tk):')
end_idx = content.find('if __name__ == "__main__":')

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + new_app_class + "\n\n" + content[end_idx:]

with open('translator.py', 'w', encoding='utf-8') as f:
    f.write(content)
