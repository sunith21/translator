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
from medical_translation import describe_translation_engine, ensure_session_models
from backend.audio_pipeline import prepare_audio_for_stt
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
    translation_engine: Optional[str] = None
    audio_filename: Optional[str] = None
    translated_audio_filename: Optional[str] = None
    stt_confidence: Optional[float] = None
    low_confidence: Optional[bool] = None
    stt_engine: Optional[str] = None
    audio_enhancement: Optional[dict] = None

class PDFRequest(BaseModel):
    transcripts: list = None

class CorrectionRequest(BaseModel):
    text: str

class TextChatRequest(BaseModel):
    patient_id: str
    text: str
    role: str = "patient"
    lang_key: str
    doctor_lang_key: str = "English"

class TTSClipRequest(BaseModel):
    text: str
    role: str
    lang_key: str
    doctor_lang_key: str = "English"

def _score_stt_quality(text: str, lang_key: str, script_consistent: bool, retried: bool, engine: str) -> tuple[float, bool]:
    """
    Heuristic STT quality score for UI feedback and retry decisions.
    """
    if not text:
        return 0.0, True

    score = 0.45
    clean_len = len((text or "").strip())
    if clean_len >= 8:
        score += 0.20
    elif clean_len >= 3:
        score += 0.10

    if script_consistent:
        score += 0.20
    else:
        score -= 0.15

    if retried:
        score -= 0.05

    if "whisper" in (engine or ""):
        score += 0.10

    score = max(0.0, min(0.99, score))
    return score, (score < 0.60)

def _normalize_whisper_detect_code(code: str, doctor_code: str, patient_code: str) -> str:
    """Konkani is often detected as Marathi by Whisper."""
    if code == "mr" and (patient_code == "gom" or doctor_code == "gom"):
        return "gom"
    return code


def _resolve_auto_role_from_probs(
    lang_probs: dict,
    doctor_code: str,
    patient_code: str,
) -> tuple[Optional[str], str, float]:
    """
    Pick doctor vs patient from Whisper language-id probabilities.
    Returns (role, top_detected_code, top_probability). role is None only when we should ignore
    the utterance (confident detection of a language outside the consultation pair).
    """
    if not lang_probs:
        return "patient", "en", 0.0

    ranked = sorted(lang_probs.items(), key=lambda kv: -kv[1])
    top_lang, top_p = ranked[0]

    if doctor_code == patient_code:
        return "patient", top_lang, float(top_p)

    doctor_best = 0.0
    patient_best = 0.0
    for lang, p in ranked:
        n = _normalize_whisper_detect_code(str(lang), doctor_code, patient_code)
        p = float(p)
        if n == doctor_code:
            doctor_best = max(doctor_best, p)
        if n == patient_code:
            patient_best = max(patient_best, p)

    if doctor_best > 0 or patient_best > 0:
        if doctor_best > patient_best:
            return "doctor", top_lang, float(top_p)
        if patient_best > doctor_best:
            return "patient", top_lang, float(top_p)
        n_top = _normalize_whisper_detect_code(str(top_lang), doctor_code, patient_code)
        if n_top == doctor_code:
            return "doctor", top_lang, float(top_p)
        if n_top == patient_code:
            return "patient", top_lang, float(top_p)
        return "patient", top_lang, float(top_p)

    if top_p < 0.55:
        return "patient", top_lang, float(top_p)

    return None, top_lang, float(top_p)

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
        return {
            "translated": req.text,
            "translation_engine": "No machine translation needed",
        }
    result = ai.translate(req.text, req.direction, req.lang_key)
    return {
        "translated": result,
        "translation_engine": describe_translation_engine(req.lang_key),
    }

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

    processed_audio = None
    stt_path = temp_path
    try:
        processed_audio = prepare_audio_for_stt(temp_path)
        stt_path = processed_audio.path
        audio_enhancement = {
            "rnnoise_applied": processed_audio.rnnoise_applied,
            "silero_vad_applied": processed_audio.vad_applied,
            "duration_seconds": round(processed_audio.duration_seconds, 2),
            "speech_seconds": round(processed_audio.speech_seconds, 2),
        }
        ensure_session_models(doctor_lang_key, lang_key)
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
            doctor_code = whisper_lang_codes.get(doctor_lang_key, "en")
            patient_code = whisper_lang_codes.get(lang_key, "hi")
            lang_probs = ai.detect_language_probs(stt_path)
            resolved, det_top, det_p = _resolve_auto_role_from_probs(
                lang_probs, doctor_code, patient_code
            )
            if resolved is None:
                print(
                    f"[Auto-detect] ignoring utterance; no doctor/patient match in Whisper "
                    f"language scores (doctor='{doctor_code}', patient='{patient_code}', "
                    f"top='{det_top}' {det_p:.0%})"
                )
                return {
                    "ignored": True,
                    "reason": "Detected a third language not selected for this consultation.",
                    "detected_language": det_top,
                    "audio_enhancement": audio_enhancement,
                }
            role = resolved
            print(f"[Auto-detect] resolved role → {role} (Whisper top={det_top} {det_p:.0%})")
        # ─────────────────────────────────────────────────────────────────────

        if role == "doctor":
            # Transcribe doctor in their chosen language
            stt_engine = f"faster-whisper-{os.environ.get('FASTER_WHISPER_MODEL', 'medium')}+"
            stt_retried = False
            if doctor_lang_key == "English":
                original = ai.whisper_stt(stt_path, "en", patient_id, role)
            else:
                forced_lang = whisper_lang_codes.get(doctor_lang_key, "en")
                try:
                    original = ai.whisper_stt(stt_path, forced_lang, patient_id, role)
                except Exception as whisper_err:
                    print(f"[Faster-Whisper STT failed for doctor, trying Sarvam fallback]: {whisper_err}")
                    original = ai.sarvam_stt(stt_path, sarvam_lang_codes.get(doctor_lang_key, "hi-IN"))
                    stt_engine = "sarvam_fallback"

                # If script looks mismatched (wrong language), retry Whisper with forced language.
                if not ai.is_language_consistent(original, doctor_lang_key):
                    print(f"[STT language mismatch] doctor transcript didn't match {doctor_lang_key}; retrying Faster-Whisper ({forced_lang})")
                    original = ai.whisper_stt(stt_path, forced_lang, patient_id, role)
                    stt_retried = True
                    stt_engine = "faster-whisper_forced_retry"

            script_ok = ai.is_language_consistent(original, doctor_lang_key)
            stt_confidence, low_confidence = _score_stt_quality(
                original, doctor_lang_key, script_ok, stt_retried, stt_engine
            )
            original = ai.correct_transcript(original, patient_id, role)

            # Translate: doctor_lang → patient_lang
            # If both are the same, no translation needed
            if doctor_lang_key == lang_key:
                translated = original
                translation_engine = "No machine translation needed"
            elif doctor_lang_key == "English" and lang_key != "English":
                # English → native patient language
                translated = ai.translate(original, "en_to_native", lang_key)
                translation_engine = describe_translation_engine(lang_key)
            elif doctor_lang_key != "English" and lang_key == "English":
                # Native doctor language → English
                translated = ai.translate(original, "native_to_en", doctor_lang_key)
                translation_engine = describe_translation_engine(doctor_lang_key)
            else:
                # Native → native in one NLLB pass (faster and more accurate than English pivot)
                translated = ai.translate_native_pair(original, doctor_lang_key, lang_key)
                translation_engine = describe_translation_engine(lang_key)
        else:
            # Patient speaking in their language
            stt_engine = f"faster-whisper-{os.environ.get('FASTER_WHISPER_MODEL', 'medium')}+"
            stt_retried = False
            if lang_key == "English":
                original = ai.whisper_stt(stt_path, "en", patient_id, role)
            else:
                # Faster-Whisper is the primary offline STT path; Sarvam is only a fallback.
                whisper_lang = whisper_lang_codes.get(lang_key, "hi")
                try:
                    original = ai.whisper_stt(stt_path, whisper_lang, patient_id, role)
                except Exception as whisper_err:
                    print(f"[Faster-Whisper STT failed, trying Sarvam fallback]: {whisper_err}")
                    original = ai.sarvam_stt(stt_path, sarvam_lang_codes.get(lang_key, "hi-IN"))
                    stt_engine = "sarvam_fallback"

                # If script looks mismatched (e.g., Malayalam text for Hindi), retry with forced Whisper lang.
                if not ai.is_language_consistent(original, lang_key):
                    forced_lang = whisper_lang_codes.get(lang_key, "hi")
                    print(f"[STT language mismatch] patient transcript didn't match {lang_key}; retrying Faster-Whisper ({forced_lang})")
                    original = ai.whisper_stt(stt_path, forced_lang, patient_id, role)
                    stt_retried = True
                    stt_engine = "faster-whisper_forced_retry"

            script_ok = ai.is_language_consistent(original, lang_key)
            stt_confidence, low_confidence = _score_stt_quality(
                original, lang_key, script_ok, stt_retried, stt_engine
            )
            original = ai.correct_transcript(original, patient_id, role)

            # Translate: patient_lang → doctor_lang
            if lang_key == doctor_lang_key:
                translated = original
                translation_engine = "No machine translation needed"
            elif lang_key == "English" and doctor_lang_key != "English":
                translated = ai.translate(original, "en_to_native", doctor_lang_key)
                translation_engine = describe_translation_engine(doctor_lang_key)
            elif lang_key != "English" and doctor_lang_key == "English":
                translated = ai.translate(original, "native_to_en", lang_key)
                translation_engine = describe_translation_engine(lang_key)
            else:
                translated = ai.translate_native_pair(original, lang_key, doctor_lang_key)
                translation_engine = describe_translation_engine(doctor_lang_key)
        
        # ── Medical transcript correction (LLM post-processing) ────────────
        # ───────────────────────────────────────────────────────────────────

        # Final guard: if transcript still looks off-language and confidence is low,
        # ignore this segment instead of polluting the conversation.
        expected_lang = doctor_lang_key if role == "doctor" else lang_key
        if low_confidence and not ai.is_language_consistent(original, expected_lang):
            return {
                "ignored": True,
                "reason": "Low-confidence transcript in unexpected language; segment skipped.",
                "detected_language": whisper_lang_codes.get(expected_lang, "unknown"),
                "stt_confidence": stt_confidence,
                "low_confidence": True,
                "stt_engine": stt_engine,
                "audio_enhancement": audio_enhancement,
            }

        # Persist this utterance so it can be replayed per-message in the UI
        utt_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"utt_{utt_ts}_{role}.wav"
        final_path = os.path.join(recordings_dir, audio_filename)
        try:
            shutil.move(stt_path, final_path)
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
            "translation_engine": translation_engine,
            "audio_filename": audio_filename,
            "audio_url": (f"/patients/{patient_id}/recordings/{audio_filename}" if audio_filename else None),
            "translated_audio_filename": translated_audio_filename,
            "translated_audio_url": (
                f"/patients/{patient_id}/recordings/{translated_audio_filename}"
                if translated_audio_filename else None
            ),
            "stt_confidence": stt_confidence,
            "low_confidence": low_confidence,
            "stt_engine": stt_engine,
            "audio_enhancement": audio_enhancement,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[STT endpoint failed] {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {e}")
    finally:
        if processed_audio and os.path.exists(processed_audio.path):
            try:
                os.remove(processed_audio.path)
            except Exception:
                pass
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

@app.post("/text_chat")
async def text_chat(req: TextChatRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    if req.role == "doctor":
        source_lang = req.doctor_lang_key
        target_lang = req.lang_key
    else:
        source_lang = req.lang_key
        target_lang = req.doctor_lang_key

    if source_lang == target_lang:
        translated = text
        translation_engine = "No machine translation needed"
    elif source_lang == "English":
        translated = ai.translate(text, "en_to_native", target_lang)
        translation_engine = describe_translation_engine(target_lang)
    elif target_lang == "English":
        translated = ai.translate(text, "native_to_en", source_lang)
        translation_engine = describe_translation_engine(source_lang)
    else:
        translated = ai.translate_native_pair(text, source_lang, target_lang)
        translation_engine = describe_translation_engine(target_lang)

    return {
        "role": req.role,
        "original": text,
        "translated": translated,
        "translation_engine": translation_engine,
    }

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

