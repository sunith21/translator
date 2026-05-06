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

class TranscriptEntry(BaseModel):
    role: str
    original: str
    translated: str

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
    lang_key: str = Form(...)
):
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"upload_{role}.wav")
    
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

        if role == "doctor":
            original = ai.whisper_stt(temp_path, "en")
            direction = "en_to_native"
        else:
            if lang_key == "English":
                original = ai.whisper_stt(temp_path, "en")
            else:
                # Try Sarvam AI first (best for Indian languages), fall back to Whisper
                try:
                    original = ai.sarvam_stt(temp_path, sarvam_lang_codes.get(lang_key, "hi-IN"))
                except Exception as sarvam_err:
                    print(f"[Sarvam STT failed, falling back to Whisper]: {sarvam_err}")
                    # Use correct Whisper language code so it doesn't misidentify the language
                    whisper_lang = whisper_lang_codes.get(lang_key, "hi")
                    original = ai.whisper_stt(temp_path, whisper_lang)
            direction = "native_to_en"
            
        if lang_key == "English":
            translated = original
        else:
            translated = ai.translate(original, direction, lang_key)
        
        return {
            "original": original,
            "translated": translated,
            "role": role
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

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

@app.get("/patients/{patient_id}/pdf")
async def get_pdf(patient_id: str):
    p = pm.get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    pdf_path = ReportGenerator.generate_pdf(patient_id, p["name"])
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=500, detail="Failed to generate PDF")
        
    return FileResponse(pdf_path, filename=f"consultation_{patient_id}.pdf")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
