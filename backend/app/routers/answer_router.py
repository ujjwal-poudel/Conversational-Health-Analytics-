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

# Import actual inference functions
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import the actual functions
import torch
from transformers import AutoTokenizer
from src.inference_service import load_artifacts, get_depression_score, set_device

# -- Modification --
# (Ujjwal) added import for semantic classifier
from src.semantic_inference import get_semantic_classifier
# -- End of Modification --

print("Real depression scoring model loaded successfully")

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

def get_next_id():
    """Get the next available ID by reading the current state from JSONL file."""
    global _id_counter
    
    # If counter is already set, just increment and return
    if _id_counter > 0:
        _id_counter += 1
        return str(_id_counter - 1)
    
    # Otherwise, read the JSONL file to find the last used ID
    try:
        if os.path.exists(USER_DATA_FILE):
            last_id = -1
            with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if 'id' in data:
                            try:
                                current_id = int(data['id'])
                                if current_id > last_id:
                                    last_id = current_id
                            except ValueError:
                                continue
            
            # Set counter to continue from the next ID
            _id_counter = last_id + 1
            print(f"Resuming from ID: {_id_counter}")
        else:
            _id_counter = 0
            
    except Exception as e:
        print(f"Error reading ID counter: {e}")
        _id_counter = 0
    
    # Return the current ID and increment for next time
    current_id = str(_id_counter)
    _id_counter += 1
    return current_id

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
    
    # -- Modified --
    # (Ujjwal) Added the semantic risk label and consistency_status
    semantic_risk_label: str
    consistency_status: str  # 'agreement', 'uncertainty', 'conflict'
    # -- End of Modification --

# Load Whisper model (tiny model for faster processing and lower resource usage)
try:
    whisper_model = whisper.load_model("tiny")
    print("Whisper tiny model loaded successfully")
except Exception as e:
    print(f"Failed to load Whisper model: {e}")
    whisper_model = None
    
"""
Function to combine the regression model with semantic similarity analysis
"""
def analyze_with_semantic_guardrails(
    turns: List[str], 
    regression_score: float
) -> Tuple[float, str, str, float, float]:
    """
    Combines the primary Regression Model score with a secondary Semantic Similarity analysis
    to act as a safety guardrail.

    Args:
        turns (List[str]): The list of conversation turns (questions and answers).
        regression_score (float): The raw score output from the main regression model.

    Returns:
        Tuple containing:
        1. Final Score (float): The score to show the user (usually the regression score).
        2. Semantic Label (str): "High Risk" or "Low Risk" based on semantic similarity.
        3. Consistency Status (str): A tag describing if models agreed or conflicted.
        4. Similarity to Healthy (float): Raw cosine similarity to the 'Healthy' prototype.
        5. Similarity to Depressed (float): Raw cosine similarity to the 'Depressed' prototype.
    """
    
    # --- STEP 1: Perform Semantic Analysis ---
    try:
        # Join the list of turns into one single string for embedding
        full_text = " ".join(turns)
        
        # Load the singleton semantic classifier
        classifier = get_semantic_classifier()
        
        # Get the prediction dictionary
        semantic_result = classifier.predict(full_text)
        
        # Extract the raw data points
        sim_0 = semantic_result["similarity_class_0"] # Similarity to 'Healthy' prototype
        sim_1 = semantic_result["similarity_class_1"] # Similarity to 'Depressed' prototype
        semantic_class = semantic_result["predicted_class"] # 0 for Healthy, 1 for Depressed
        label = semantic_result["predicted_label"] # Text label for display
        
        # Calculate "Confidence Margin":
        # If sim_0 is 0.45 and sim_1 is 0.46, the margin is 0.01 (Very Low Confidence).
        # If sim_0 is 0.20 and sim_1 is 0.70, the margin is 0.50 (Very High Confidence).
        confidence_margin = abs(sim_1 - sim_0)
        
    except Exception as e:
        # ERROR HANDLING: If the semantic check fails (e.g., file missing),
        # we do NOT stop the app. We fail safely by returning the regression score.
        print(f"Semantic Analysis Failed: {e}")
        return regression_score, "N/A", "semantic_failed", 0.0, 0.0

    # --- STEP 2: Standardize the Regression Output ---
    # We convert the continuous regression score (e.g., 8.5 or 12.3) into a binary class
    # using the standard PHQ-8 threshold of 10.
    regression_class = 1 if regression_score >= 10 else 0
    
    # Default status
    status = "processed"

    # --- STEP 3: The Logic Gates (Guardrails) ---
    
    # GATE 1: Low Confidence Check
    # If the semantic model is "unsure" (the text is ambiguous and equally similar to both),
    # we should NOT let it override or flag the main model.
    # We trust the Regression Model completely in this case.
    if confidence_margin < 0.05:
        status = "agreement_low_confidence"
        return regression_score, label, status, sim_0, sim_1

    # GATE 2: Agreement Check
    # Ideally, both models agree (e.g., both say "High Risk" or both say "Low Risk").
    # This validates our prediction and gives us high confidence.
    if semantic_class == regression_class:
        status = "strong_agreement"
        return regression_score, label, status, sim_0, sim_1

    # GATE 3: Handling Conflicts (The Models Disagree)
    
    # Case A: Semantic says DEPRESSED (1), but Regression says HEALTHY (0)
    # This is a potential FALSE NEGATIVE. 
    # The user's words are semantically close to "depressed" text, 
    # but the regression model calculated a low score (e.g., 8 or 9).
    if semantic_class == 1 and regression_class == 0:
        status = "conflict_potential_false_negative"
        # We return the Regression Score (because it's the primary clinical tool),
        # but the 'status' flag warns us to review this case manually or treat it with caution.
        return regression_score, "High Risk (Semantic)", status, sim_0, sim_1

    # Case B: Semantic says HEALTHY (0), but Regression says DEPRESSED (1)
    # This is a potential FALSE POSITIVE.
    # The user sounds "normal" semantically, but scored high on the clinical scale.
    # In medical contexts, we prefer False Positives over False Negatives (better safe than sorry).
    # We stick with the High Score to ensure the user gets help if needed.
    if semantic_class == 0 and regression_class == 1:
        status = "conflict_potential_false_positive"
        return regression_score, "Low Risk (Semantic)", status, sim_0, sim_1

    # Fallback return (should theoretically not be reached given logic above)
    return regression_score, label, status, sim_0, sim_1
    
def initialize_model():
    """Initialize the depression scoring model and tokenizer."""
    global _model, _tokenizer, _device
    
    if _model is not None:
        return  # Already initialized
    
    try:
        print("Initializing depression scoring model...")
        _device = set_device()
        
        # Try different possible model paths
        possible_paths = [
            "/Volumes/MACBACKUP/models/saved_models/robert_multilabel_no-regression_/model_2_15.pt",
            "models/robert_multilabel_no-regression_/model_2_15.pt",
            "model_2_15.pt"
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError(f"Model file not found. Tried paths: {possible_paths}")
        
        _model, _tokenizer = load_artifacts(
            model_path=model_path,
            tokenizer_name="sentence-transformers/all-distilroberta-v1",
            device=_device
        )
        print("Depression scoring model initialized successfully")
        
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        raise e  # Re-raise the exception - no fallback

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
    # Initialize model if not already done
    if _model is None:
        initialize_model()
    
    if _model is None or _tokenizer is None or _device is None:
        raise HTTPException(status_code=500, detail="Model not available")
    
    try:
        # Transform question/answer data to turns format
        turns = transform_to_turns(request)
        
        # Get depression score using the actual function
        raw_score = get_depression_score(
            transcript_turns=turns,
            model=_model,
            tokenizer=_tokenizer,
            device=_device,
            turn_batch_size=16
        )
        
        # --- Modification ---
        # (Ujjwal) Applying semantic guardrail function
        
        final_score, sem_label, status, sim_0, sim_1 = analyze_with_semantic_guardrails(turns, raw_score)
        
        # DETAILED TERMINAL LOGGING
        print("\n" + "#"*60)
        print(" FINAL DECISION REPORT ".center(60, "#"))
        print("#"*60)
        print(f"1. RAW REGRESSION MODEL:")
        print(f"   Score: {raw_score:.4f}")
        print(f"   Class: {'Depressed (>=10)' if raw_score >= 10 else 'Healthy (<10)'}")
        print("-" * 60)
        print(f"2. SEMANTIC GUARDRAIL:")
        print(f"   Healthy Similarity:   {sim_0:.4f}")
        print(f"   Depressed Similarity: {sim_1:.4f}")
        print(f"   Semantic Label:       {sem_label}")
        print("-" * 60)
        print(f"3. CONSISTENCY CHECK:")
        print(f"   Status: {status.upper()}")
        
        if "conflict" in status:
             print(f" WARNING: Models Disagree! Returning Regression Score with Flag.")
        elif "agreement" in status:
             print(f" SUCCESS: Models Agree.")
        
        print("-" * 60)
        print(f"   >>> FINAL RETURNED SCORE: {final_score:.4f}")
        print("#"*60 + "\n")
        
        # Get persistent ID (continues from previous session)
        current_id = get_next_id()
        save_to_jsonl(current_id, turns, final_score)
        
        # Returns full prediction
        return PredictionResponse(
            prediction=final_score,
            semantic_risk_label=sem_label,
            consistency_status=status
        )
        # --- End of Modification ---
        
    except Exception as e:
        print(f"Error processing text answer: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")