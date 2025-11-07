import os
import shutil
import uuid
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from .. import models, database
from pydantic import BaseModel
import whisper

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load Whisper model (tiny model for faster processing and lower resource usage)
try:
    whisper_model = whisper.load_model("tiny")
    print("Whisper tiny model loaded successfully")
except Exception as e:
    print(f"Failed to load Whisper model: {e}")
    whisper_model = None

class SubmitAnswerRequest(BaseModel):
    name: str
    question_id: int

@router.post("/submit-answer")
def submit_answer(
    name: str = Form(...),
    question_id: int = Form(...),
    audio: UploadFile = File(...),
    db: Session = Depends(database.get_db)
):
    # Find or create user
    user = db.query(models.User).filter(models.User.name == name).first()
    if not user:
        user = models.User(id=str(uuid.uuid4()), name=name, is_admin=False)
        db.add(user)
        db.commit()
        db.refresh(user)

    # Save audio file
    file_extension = os.path.splitext(audio.filename)[1]
    file_path = os.path.join(UPLOAD_DIR, f"{user.id}_{question_id}_{uuid.uuid4()}{file_extension}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    # Transcribe audio using Whisper
    transcript = "Transcription not available - Whisper model not loaded"
    if whisper_model:
        try:
            # Use Whisper to transcribe the audio file
            result = whisper_model.transcribe(file_path)
            transcript = result["text"].strip()
            print(f"Whisper transcription completed: {transcript}")
        except Exception as e:
            print(f"Whisper transcription error: {e}")
            transcript = f"Transcription failed: {str(e)}"
    else:
        transcript = "Whisper model not available - transcription disabled"

    # Save answer to DB
    answer = models.Answer(
        user_uuid=user.id,
        question_id=question_id,
        audio_file_path=file_path,
        transcript_text=transcript
    )
    db.add(answer)
    db.commit()

    return {"message": "Answer submitted successfully", "user_uuid": user.id, "transcript": transcript}

@router.get("/audio/{filename}")
def get_audio(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="audio/wav")