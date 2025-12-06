from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for inference_service
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
sys.path.append(src_dir)

import inference_service
from src.semantic_inference import get_semantic_classifier
from app.database import engine
from app import models
from app.routers import auth_router, question_router, answer_router, admin_router
from app.api.v1 import chat_router, audio_chat_router
from app.conversation.engine import ConversationEngine

# --- Configuration ---
TEMPLATES_PATH = os.path.join(current_dir, "app/conversation/data")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize Chatbot Engine
    print("Initializing Conversation Engine (Gemini)...")
    try:
        conversation_engine = ConversationEngine(templates_base_path=TEMPLATES_PATH)
        app.state.conversation_engine = conversation_engine
        app.state.templates_path = TEMPLATES_PATH
        print("Conversation Engine initialized.")
    except Exception as e:
        print(f"Failed to initialize Conversation Engine: {e}")

    # 2. Initialize Depression Model
    print("Initializing Depression Model...")
    device = inference_service.set_device()
    
    # Try different possible model paths
    # Best model is model_2_13 (epoch 14) with MAE=4.73
    possible_paths = [
        "/Volumes/MACBACKUP/models/saved_models/robert_multilabel_no-regression_/model_2_13.pt",
        "models/robert_multilabel_no-regression_/model_2_13.pt",
        "model_2_13.pt"
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
            
    if model_path:
        model, tokenizer = inference_service.load_artifacts(
            model_path=model_path,
            tokenizer_name="sentence-transformers/all-distilroberta-v1",
            device=device
        )
        app.state.depression_model = model
        app.state.depression_tokenizer = tokenizer
        app.state.device = device
        print("Depression Model loaded.")
    else:
        print("WARNING: Depression model file not found. Scoring will be disabled.")

    # 3. Initialize Semantic Classifier
    print("Initializing Semantic Classifier...")
    try:
        get_semantic_classifier()
        print("Semantic Classifier loaded.")
    except Exception as e:
        print(f"WARNING: Failed to load Semantic Classifier: {e}")

    # 4. Initialize Audio Inference Service (for multimodal fusion)
    print("Initializing Audio Inference Service...")
    try:
        from src.audio_inference_service import AudioInferenceService
        audio_service = AudioInferenceService()
        audio_service.load_models()
        app.state.audio_service = audio_service
        print("Audio Inference Service loaded successfully.")
    except FileNotFoundError as e:
        print(f"WARNING: Audio models not found (harddrive may not be mounted): {e}")
        app.state.audio_service = None
    except Exception as e:
        print(f"WARNING: Failed to load Audio Inference Service: {e}")
        app.state.audio_service = None

    yield
    # Clean up resources if needed
    print("Shutting down...")

models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Audio Question App", 
    description="API for audio question submissions with transcription",
    lifespan=lifespan
)

# Mount audio directory for static access
os.makedirs("audio_data", exist_ok=True)
app.mount("/audio", StaticFiles(directory="audio_data"), name="audio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router, prefix="/auth", tags=["auth"])
app.include_router(question_router.router, tags=["questions"])
app.include_router(answer_router.router, tags=["answers"])
app.include_router(admin_router.router, prefix="/admin", tags=["admin"])
app.include_router(chat_router.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(audio_chat_router.router, prefix="/api/v1", tags=["audio"])

@app.get("/")
def read_root():
    return {"message": "Audio Question App API"}