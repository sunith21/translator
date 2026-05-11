import os
import shutil
import tempfile
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datetime import datetime
from dotenv import load_dotenv
from backend.services import PatientManager, AIService, ReportGenerator, EmailService

# Load .env so SARVAM_API_KEY is available
load_dotenv()


app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pm = PatientManager()
ai = AIService()

class PatientCreate(BaseModel):
    id: str
    name: str
    email: str = ""

class TranslationRequest(BaseModel):
    text: str
    direction: str
    lang_key: str
    doctor_lang_key: str = "English"

class TranscriptEntry(BaseModel):
    role: str
    original: str
    translated: str
    audio_filename: Optional[str] = None
    translated_audio_filename: Optional[str] = None

class PDFRequest(BaseModel):
    transcripts: list = None

class CorrectionRequest(BaseModel):
    text: str

class TTSClipRequest(BaseModel):
    text: str
    role: str
    lang_key: str
    doctor_lang_key: str = "English"

@app.get("/patients/recent")
async def get_recent_patients():
    return pm.get_recent_patients()

@app.get("/patients/search")
async def search_patients(q: str):
    return pm.search_patients(q)

@app.post("/patients")
async def create_patient(patient: PatientCreate):
    pm.add_patient(patient.id, patient.name, patient.email)
    return {"status": "success"}

@app.get("/patients/{patient_id}")
async def get_patient(patient_id: str):
    p = pm.get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    pm.update_last_visit(patient_id)
    return p

@app.get("/patients/{patient_id}/transcripts")
async def get_transcripts(patient_id: str):
    path = os.path.join("data", "patients", patient_id, "transcripts.json")
    if not os.path.exists(path):
        return []
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/patients/{patient_id}/clear_chat")
async def clear_chat(patient_id: str):
    patient = pm.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
        
    pdf_path = ReportGenerator.archive_chat(patient_id, patient["name"])
    if not pdf_path:
        # No chat to archive, just clear if needed
        path = os.path.join("data", "patients", patient_id, "transcripts.json")
        if os.path.exists(path):
            import json
            with open(path, "w", encoding="utf-8") as f:
                json.dump([], f)
        return {"status": "cleared"}
        
    return {"status": "archived"}

@app.post("/translate")
async def translate(req: TranslationRequest):
    if req.lang_key == "English":
        return {"translated": req.text}
    result = ai.translate(req.text, req.direction, req.lang_key)
    return {"translated": result}

@app.post("/stt")
async def speech_to_text(
    file: UploadFile = File(...),
    role: str = Form(...),
    lang_key: str = Form(...),
    patient_id: str = Form(...),
    doctor_lang_key: str = Form(default="English")
):
    # Ensure patient recordings directory exists
    recordings_dir = os.path.join("data", "patients", patient_id, "recordings")
    os.makedirs(recordings_dir, exist_ok=True)

    # Save upload to a true temp file first; we will persist it as an utterance
    # only after we've resolved the final role (doctor/patient) in auto mode.
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_path = tmp.name
    tmp.close()
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Sarvam uses BCP-47 codes (hi-IN), Whisper uses ISO 639-1 (hi)
        sarvam_lang_codes = {
            "Hindi (हिन्दी)":   "hi-IN",
            "Kannada (ಕನ್ನಡ)": "kn-IN",
            "Marathi (मराठी)":  "mr-IN",
            "Bengali (বাংলা)":  "bn-IN",
            "Malayalam (മലയാളം)": "ml-IN",
            "Tamil (தமிழ்)": "ta-IN",
            "Konkani (कोंकणी)": "gom-IN",
            "English": "en-IN"
        }
        whisper_lang_codes = {
            "Hindi (हिन्दी)":   "hi",
            "Kannada (ಕನ್ನಡ)": "kn",
            "Marathi (मराठी)":  "mr",
            "Bengali (বাংলা)":  "bn",
            "Malayalam (മലയാളം)": "ml",
            "Tamil (தமிழ்)": "ta",
            "Konkani (कोंकणी)": "gom",
            "English": "en"
        }

        # ── Auto-detect speaker from spoken language ──────────────────────────
        if role == "auto":
            detected_code = ai.detect_language(temp_path)
            doctor_code   = whisper_lang_codes.get(doctor_lang_key, "en")
            patient_code  = whisper_lang_codes.get(lang_key, "hi")

            # Konkani is often detected as Marathi by Whisper — correct for it
            if detected_code == "mr" and patient_code == "gom":
                detected_code = "gom"
            if detected_code == "mr" and doctor_code == "gom":
                detected_code = "gom"

            if detected_code == doctor_code:
                role = "doctor"
            elif detected_code == patient_code:
                role = "patient"
            else:
                # Ambiguous — default to patient
                print(f"[Auto-detect] '{detected_code}' matched neither "
                      f"doctor '{doctor_code}' nor patient '{patient_code}'. Defaulting to patient.")
                role = "patient"
            print(f"[Auto-detect] resolved role → {role}")
        # ─────────────────────────────────────────────────────────────────────

        if role == "doctor":
            # Transcribe doctor in their chosen language
            if doctor_lang_key == "English":
                original = ai.whisper_stt(temp_path, "en")
            else:
                try:
                    original = ai.sarvam_stt(temp_path, sarvam_lang_codes.get(doctor_lang_key, "hi-IN"))
                except Exception as sarvam_err:
                    print(f"[Sarvam STT failed for doctor, falling back to Whisper]: {sarvam_err}")
                    original = ai.whisper_stt(temp_path, whisper_lang_codes.get(doctor_lang_key, "en"))

                # If script looks mismatched (wrong language), retry Whisper with forced language.
                if not ai.is_language_consistent(original, doctor_lang_key):
                    forced_lang = whisper_lang_codes.get(doctor_lang_key, "en")
                    print(f"[STT language mismatch] doctor transcript didn't match {doctor_lang_key}; retrying Whisper ({forced_lang})")
                    original = ai.whisper_stt(temp_path, forced_lang)

            # Translate: doctor_lang → patient_lang
            # If both are the same, no translation needed
            if doctor_lang_key == lang_key:
                translated = original
            elif doctor_lang_key == "English" and lang_key != "English":
                # English → native patient language
                translated = ai.translate(original, "en_to_native", lang_key)
            elif doctor_lang_key != "English" and lang_key == "English":
                # Native doctor language → English
                translated = ai.translate(original, "native_to_en", doctor_lang_key)
            else:
                # Native language A → English → Native language B (pivot translation)
                english_pivot = ai.translate(original, "native_to_en", doctor_lang_key)
                translated = ai.translate(english_pivot, "en_to_native", lang_key)
        else:
            # Patient speaking in their language
            if lang_key == "English":
                original = ai.whisper_stt(temp_path, "en")
            else:
                # Try Sarvam AI first (best for Indian languages), fall back to Whisper
                try:
                    original = ai.sarvam_stt(temp_path, sarvam_lang_codes.get(lang_key, "hi-IN"))
                except Exception as sarvam_err:
                    print(f"[Sarvam STT failed, falling back to Whisper]: {sarvam_err}")
                    whisper_lang = whisper_lang_codes.get(lang_key, "hi")
                    original = ai.whisper_stt(temp_path, whisper_lang)

                # If script looks mismatched (e.g., Malayalam text for Hindi), retry with forced Whisper lang.
                if not ai.is_language_consistent(original, lang_key):
                    forced_lang = whisper_lang_codes.get(lang_key, "hi")
                    print(f"[STT language mismatch] patient transcript didn't match {lang_key}; retrying Whisper ({forced_lang})")
                    original = ai.whisper_stt(temp_path, forced_lang)

            # Translate: patient_lang → doctor_lang
            if lang_key == doctor_lang_key:
                translated = original
            elif lang_key == "English" and doctor_lang_key != "English":
                translated = ai.translate(original, "en_to_native", doctor_lang_key)
            elif lang_key != "English" and doctor_lang_key == "English":
                translated = ai.translate(original, "native_to_en", lang_key)
            else:
                # Native language A → English → Native language B (pivot translation)
                english_pivot = ai.translate(original, "native_to_en", lang_key)
                translated = ai.translate(english_pivot, "en_to_native", doctor_lang_key)
        
        # ── Medical transcript correction (LLM post-processing) ────────────
        original   = ai.correct_transcript(original)
        # ───────────────────────────────────────────────────────────────────

        # Persist this utterance so it can be replayed per-message in the UI
        utt_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"utt_{utt_ts}_{role}.wav"
        final_path = os.path.join(recordings_dir, audio_filename)
        try:
            shutil.move(temp_path, final_path)
        except Exception:
            # If move fails for any reason, fall back to leaving it as temp
            audio_filename = None

        # Generate translated playback audio (AI voice) and save it.
        translated_audio_filename = None
        try:
            target_tts_lang = lang_key if role == "doctor" else doctor_lang_key
            tts_bytes, content_type = ai.generate_tts(translated, target_tts_lang)
            ext = ".mp3" if "mpeg" in (content_type or "").lower() else ".wav"
            translated_audio_filename = f"tts_{utt_ts}_{role}{ext}"
            translated_audio_path = os.path.join(recordings_dir, translated_audio_filename)
            with open(translated_audio_path, "wb") as f:
                f.write(tts_bytes)
        except Exception as tts_err:
            print(f"[TTS generation failed] {tts_err}")

        return {
            "original":   original,
            "translated": translated,
            "role":       role,
            "audio_filename": audio_filename,
            "audio_url": (f"/patients/{patient_id}/recordings/{audio_filename}" if audio_filename else None),
            "translated_audio_filename": translated_audio_filename,
            "translated_audio_url": (
                f"/patients/{patient_id}/recordings/{translated_audio_filename}"
                if translated_audio_filename else None
            ),
        }
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@app.post("/correct_transcript")
async def correct_transcript_endpoint(req: CorrectionRequest):
    """Standalone endpoint: run any text through the medical correction LLM."""
    corrected = ai.correct_transcript(req.text)
    return {"original": req.text, "corrected": corrected}

@app.post("/patients/{patient_id}/append_transcript")
async def append_transcript(patient_id: str, entry: TranscriptEntry):
    path = os.path.join("data", "patients", patient_id, "transcripts.json")
    import json
    data = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    
    record = entry.model_dump()
    record["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data.append(record)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        
    return {"status": "success"}

@app.post("/patients/{patient_id}/tts_clip")
async def create_tts_clip(patient_id: str, req: TTSClipRequest):
    """Create and persist a translated TTS clip for one utterance."""
    p = pm.get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")

    rec_dir = os.path.join("data", "patients", patient_id, "recordings")
    os.makedirs(rec_dir, exist_ok=True)

    try:
        target_tts_lang = req.lang_key if req.role == "doctor" else req.doctor_lang_key
        tts_bytes, content_type = ai.generate_tts(req.text, target_tts_lang)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = ".mp3" if "mpeg" in (content_type or "").lower() else ".wav"
        filename = f"tts_{ts}_{req.role}{ext}"
        path = os.path.join(rec_dir, filename)
        with open(path, "wb") as f:
            f.write(tts_bytes)
        return {
            "translated_audio_filename": filename,
            "translated_audio_url": f"/patients/{patient_id}/recordings/{filename}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS clip generation failed: {e}")

@app.get("/patients/{patient_id}/recordings/{filename}")
async def get_recording(patient_id: str, filename: str):
    """Serve a saved per-utterance or session recording for playback in the UI."""
    # Basic path traversal protection: only allow simple filenames
    if filename != os.path.basename(filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    rec_path = os.path.join("data", "patients", patient_id, "recordings", filename)
    if not os.path.exists(rec_path):
        raise HTTPException(status_code=404, detail="Recording not found")
    return FileResponse(rec_path, filename=filename)

@app.post("/patients/{patient_id}/pdf")
async def get_pdf_with_data(patient_id: str, req: PDFRequest):
    """Generate PDF from the live transcript data passed by the frontend."""
    p = pm.get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")

    pdf_path = ReportGenerator.generate_pdf(patient_id, p["name"], transcripts=req.transcripts)
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=500, detail="Failed to generate PDF")

    return FileResponse(pdf_path, filename=f"consultation_{patient_id}.pdf")

@app.post("/patients/{patient_id}/email_report")
async def email_report(patient_id: str, req: PDFRequest):
    """Generate PDF and email it to the patient."""
    p = pm.get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    if not p.get("email"):
        raise HTTPException(status_code=400, detail="No email on file for this patient")

    pdf_path = ReportGenerator.generate_pdf(patient_id, p["name"], transcripts=req.transcripts)
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=500, detail="Failed to generate PDF")

    try:
        res = EmailService.send_report(p["email"], p["name"], pdf_path)
        return res
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send email: {e}")

@app.get("/patients/{patient_id}/pdf")
async def get_pdf(patient_id: str):
    """Generate PDF from saved transcripts.json on disk."""
    p = pm.get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")

    pdf_path = ReportGenerator.generate_pdf(patient_id, p["name"])
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=500, detail="Failed to generate PDF")

    return FileResponse(pdf_path, filename=f"consultation_{patient_id}.pdf")

@app.get("/patients/{patient_id}/archives")
async def list_archives(patient_id: str):
    p = pm.get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    return ReportGenerator.list_archives(patient_id)

@app.get("/patients/{patient_id}/archives/{filename}")
async def download_archive(patient_id: str, filename: str):
    archive_path = os.path.join("data", "patients", patient_id, "history", filename)
    if not os.path.exists(archive_path):
        raise HTTPException(status_code=404, detail="Archive not found")
    return FileResponse(archive_path, filename=filename)


@app.get("/patients/{patient_id}/recordings")
async def list_recordings(patient_id: str):
    """List saved recordings (per-utterance + any full-session files) for a patient."""
    p = pm.get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")

    rec_dir = os.path.join("data", "patients", patient_id, "recordings")
    if not os.path.exists(rec_dir):
        return []

    items = []
    for name in sorted(os.listdir(rec_dir), reverse=True):
        if not (name.lower().endswith(".wav") or name.lower().endswith(".mp3") or name.lower().endswith(".ogg") or name.lower().endswith(".webm")):
            continue
        full = os.path.join(rec_dir, name)
        if not os.path.isfile(full):
            continue
        try:
            stat = os.stat(full)
            size_kb = round(stat.st_size / 1024, 1)
        except Exception:
            size_kb = None

        kind = "recording"
        role = None
        if name.startswith("utt_"):
            kind = "utterance"
            # utt_YYYYMMDD_HHMMSS_role.wav
            parts = name.split("_")
            if len(parts) >= 4:
                role = parts[3].split(".")[0]
        elif name.startswith("tts_"):
            kind = "translated_tts"
            # tts_YYYYMMDD_HHMMSS_role.wav|mp3
            parts = name.split("_")
            if len(parts) >= 4:
                role = parts[3].split(".")[0]
        elif name.startswith("session_"):
            kind = "session"

        items.append({
            "filename": name,
            "kind": kind,
            "role": role,
            "size_kb": size_kb,
            "url": f"/patients/{patient_id}/recordings/{name}",
        })

    return items

