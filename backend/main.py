from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
from app.api.v1 import chat_router
from app.conversation.engine import ConversationEngine

# --- Configuration ---
TEMPLATES_PATH = os.path.join(current_dir, "app/conversation/data")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize Chatbot Engine
    print("Initializing Conversation Engine (Groq)...")
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
            
    if model_path:
        print(f"Loading model from: {model_path}...")
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

    yield
    # Clean up resources if needed
    print("Shutting down...")

models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Audio Question App", 
    description="API for audio question submissions with transcription",
    lifespan=lifespan
)

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

@app.get("/")
def read_root():
    return {"message": "Audio Question App API"}