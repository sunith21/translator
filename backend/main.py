import os
import shutil
import tempfile
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datetime import datetime
from dotenv import load_dotenv
from backend.services import PatientManager, AIService, ReportGenerator

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

class TranslationRequest(BaseModel):
    text: str
    direction: str
    lang_key: str
    doctor_lang_key: str = "English"

class TranscriptEntry(BaseModel):
    role: str
    original: str
    translated: str

class PDFRequest(BaseModel):
    transcripts: list = None

class CorrectionRequest(BaseModel):
    text: str

@app.get("/patients/recent")
async def get_recent_patients():
    return pm.get_recent_patients()

@app.get("/patients/search")
async def search_patients(q: str):
    return pm.search_patients(q)

@app.post("/patients")
async def create_patient(patient: PatientCreate):
    pm.add_patient(patient.id, patient.name)
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

    timestamp = int(datetime.now().timestamp())
    temp_path = os.path.join(recordings_dir, f"upload_{role}_{timestamp}.wav")

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

        return {
            "original":   original,
            "translated": translated,
            "role":       role
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

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

