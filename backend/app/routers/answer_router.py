import os
import shutil
import uuid
import json
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from .. import models, database
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import whisper

# For now, use a dummy function to avoid dependency issues
def get_depression_score_dummy(transcript_turns, model=None, tokenizer=None, device=None, turn_batch_size=16):
    """
    Dummy depression scoring function for testing.
    Returns a random score between 5-25 for demonstration.
    """
    import random
    return random.uniform(5.0, 25.0)

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# User data logging
USER_DATA_DIR = "user_data"
os.makedirs(USER_DATA_DIR, exist_ok=True)
USER_DATA_FILE = os.path.join(USER_DATA_DIR, "user_data.jsonl")

# Global variables for model initialization
_model = None
_tokenizer = None
_device = None
_id_counter = 0

# Define the input models
class SubQuestion(BaseModel):
    id: str
    text: str = Field(alias="text")  # Keeping "text" as requested in the example
    answer: str

class Question(BaseModel):
    id: int
    mainQuestion: str
    answer: str
    subQuestions: List[SubQuestion]

class QuestionAnswerRequest(BaseModel):
    questions: List[Question]

class PredictionResponse(BaseModel):
    prediction: float

# Load Whisper model (tiny model for faster processing and lower resource usage)
try:
    whisper_model = whisper.load_model("tiny")
    print("Whisper tiny model loaded successfully")
except Exception as e:
    print(f"Failed to load Whisper model: {e}")
    whisper_model = None

def initialize_model():
    """Initialize dummy depression scoring (for testing)."""
    global _model, _tokenizer, _device
    
    if _model is not None:
        return  # Already initialized
    
    try:
        print("Initializing dummy depression scoring model for testing...")
        # Set dummy values for testing
        _model = "dummy_model"
        _tokenizer = "dummy_tokenizer"
        _device = "cpu"
        print("Dummy model initialized successfully")
        
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        _model = None
        _tokenizer = None
        _device = None

def transform_to_turns(questions_data: QuestionAnswerRequest) -> List[str]:
    """Transform question/answer JSON to turns format."""
    turns = []
    
    for question in questions_data.questions:
        # Add main question and answer
        turns.append(question.mainQuestion)
        turns.append(question.answer)
        
        # Add sub-questions and answers
        for sub_q in question.subQuestions:
            turns.append(sub_q.text)  # Using "text" as specified
            turns.append(sub_q.answer)
    
    return turns

def save_to_jsonl(id_val: str, turns: List[str], label: float):
    """Save data to JSONL file."""
    try:
        data = {
            "id": id_val,
            "turns": turns,
            "labels": [label]
        }
        
        with open(USER_DATA_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data) + '\n')
            
    except Exception as e:
        print(f"Failed to save to JSONL: {e}")

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
@router.post("/submit-text-answer", response_model=PredictionResponse)
def submit_text_answer(
    request: QuestionAnswerRequest,
    db: Session = Depends(database.get_db)
):
    """
    Submit text-based answers and get depression score prediction.
    """
    global _id_counter
    
    # Initialize model if not already done
    if _model is None:
        initialize_model()
    
    if _model is None or _tokenizer is None or _device is None:
        raise HTTPException(status_code=500, detail="Model not available")
    
    try:
        # Transform question/answer data to turns format
        turns = transform_to_turns(request)
        
        # Get depression score (using dummy function for testing)
        score = get_depression_score_dummy(
            transcript_turns=turns,
            model=_model,
            tokenizer=_tokenizer,
            device=_device,
            turn_batch_size=16
        )
        
        # Save to JSONL
        current_id = str(_id_counter)
        save_to_jsonl(current_id, turns, score)
        
        # Increment counter
        _id_counter += 1
        
        # Return only prediction
        return PredictionResponse(prediction=score)
        
    except Exception as e:
        print(f"Error processing text answer: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")