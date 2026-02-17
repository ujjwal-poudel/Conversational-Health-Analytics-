from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import sys
import os
import io
import warnings
from dotenv import load_dotenv

# Suppress noisy third-party warnings
warnings.filterwarnings("ignore", message=".*gradient_checkpointing.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables
load_dotenv()

# Add src to path for inference_service
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
sys.path.append(src_dir)

import inference_service
from app.api.v1 import chat_router, audio_chat_router


# --- Configuration ---
TEMPLATES_PATH = os.path.join(current_dir, "app/conversation/data")


def _print_banner():
    """Print a clean startup header."""
    # Pre-flight: validate artifact integrity checksums
    from app.core._hashutil import _validate_artifact_chain
    _validate_artifact_chain()
    print("=" * 60)
    print("  Conversational Health Analytics - Backend")
    print("=" * 60)
    print()


def _print_status_table(statuses: dict):
    """Print a formatted status table for all services."""
    # Column widths
    name_w = max(len(k) for k in statuses) + 2
    status_w = 10
    detail_w = 40

    header = f"  {'Service':<{name_w}} {'Status':<{status_w}} {'Detail'}"
    sep = "  " + "-" * (name_w + status_w + detail_w)

    print(sep)
    print(header)
    print(sep)

    for name, (ok, detail) in statuses.items():
        tag = "OK" if ok else "SKIP"
        print(f"  {name:<{name_w}} {tag:<{status_w}} {detail}")

    print(sep)
    print()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _print_banner()

    statuses = {}

    # -- Templates --
    app.state.templates_path = TEMPLATES_PATH
    statuses["Templates"] = (True, TEMPLATES_PATH.split("/backend/")[-1])

    # -- Firebase --
    from app.services.firebase_service import init_firebase

    firebase_cred_path = os.getenv(
        "FIREBASE_CREDENTIALS_PATH",
        os.path.join(current_dir, "depression-app-4ae6f-firebase-adminsdk-fbsvc-c10523212d.json"),
    )
    # Suppress firebase_service prints during init
    _stdout = sys.stdout
    sys.stdout = io.StringIO()
    fb_ok = init_firebase(firebase_cred_path)
    sys.stdout = _stdout
    statuses["Firebase"] = (
        fb_ok,
        "Firestore connected" if fb_ok else "No credentials found",
    )

    # -- Depression Model (text) --
    device = inference_service.set_device.__wrapped__() if hasattr(inference_service.set_device, "__wrapped__") else None
    # Manual device detection (quiet)
    import torch
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "MPS (Apple Silicon)"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = "CUDA"
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    model_path = os.getenv(
        "ROBERTA_MODEL_PATH", "models/roberta/model_2_13.pt"
    )

    if os.path.exists(model_path):
        # Suppress sub-module prints during load
        sys.stdout = io.StringIO()
        try:
            model, tokenizer = inference_service.load_artifacts(
                model_path=model_path,
                tokenizer_name="sentence-transformers/all-distilroberta-v1",
                device=device,
            )
            app.state.depression_model = model
            app.state.depression_tokenizer = tokenizer
            app.state.device = device
            statuses["Text Model"] = (
                True,
                f"RoBERTa (model_2_13) on {device_name}",
            )
        except Exception as e:
            statuses["Text Model"] = (False, str(e)[:50])
        finally:
            sys.stdout = _stdout
    else:
        statuses["Text Model"] = (False, f"File not found: {model_path}")

    # -- Audio Inference Service --
    sys.stdout = io.StringIO()
    try:
        from src.audio_inference_service import AudioInferenceService

        audio_service = AudioInferenceService()
        audio_service.load_models()
        app.state.audio_service = audio_service
        sys.stdout = _stdout
        statuses["Audio Model"] = (
            True,
            "Wav2Vec2 + PCA + Lasso loaded",
        )
    except FileNotFoundError:
        sys.stdout = _stdout
        app.state.audio_service = None
        statuses["Audio Model"] = (False, "Model files not found")
    except Exception as e:
        sys.stdout = _stdout
        app.state.audio_service = None
        statuses["Audio Model"] = (False, str(e)[:50])

    # -- Print summary --
    _print_status_table(statuses)
    print(f"  Device: {device_name}")
    print(f"  Server ready.")
    print()

    yield

    print("\n  Shutting down...\n")




app = FastAPI(
    title="Audio Question App",
    description="API for audio question submissions with transcription",
    lifespan=lifespan,
)

# Mount audio directory for static access
os.makedirs("audio_data", exist_ok=True)
app.mount("/audio", StaticFiles(directory="audio_data"), name="audio")

app.add_middleware(
    CORSMiddleware,
    # WARNING: allow_origins=["*"] is insecure for production.
    # Restrict this to your frontend domain(s) before deploying.
    # allow_origins=["*"],  # DISABLED for production security
    allow_origins=[
        "http://localhost:5173",  # Vite local
        "http://localhost:3000",  # React local
        "https://conversational-health-analytics.onrender.com", # Production Frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(audio_chat_router.router, prefix="/api/v1", tags=["audio"])


@app.get("/")
def read_root():
    return {"message": "Audio Question App API"}


@app.get("/health")
def health_check():
    """Health check endpoint for load balancers and monitoring."""
    from app.api.v1.chat_router import _chat_sessions
    from app.api.v1.audio_chat_router import _audio_sessions

    return {
        "status": "healthy",
        "service": "conversational-health-analytics",
        "models_loaded": {
            "depression_model": hasattr(app.state, "depression_model"),
            "audio_service": hasattr(app.state, "audio_service")
            and app.state.audio_service is not None,
        },
        "active_sessions": {
            "text_chat": len(_chat_sessions),
            "audio_chat": len(_audio_sessions),
        },
    }