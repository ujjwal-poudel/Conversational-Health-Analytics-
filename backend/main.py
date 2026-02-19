from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import sys
import os
import asyncio
import logging
import warnings
from dotenv import load_dotenv
from slowapi.errors import RateLimitExceeded

# Suppress noisy third-party warnings
warnings.filterwarnings("ignore", message=".*gradient_checkpointing.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Logging Configuration ────────────────────────────────────────────
# This is the root config — it controls what every logger in the app outputs.
#
# How it works:
#   1. Each file creates its own logger: logger = logging.getLogger(__name__)
#   2. Loggers inherit the root config set here (level, format, etc.)
#   3. You can silence noisy modules by raising their level:
#        logging.getLogger("noisy.module").setLevel(logging.WARNING)
#
# Levels (lowest → highest):
#   DEBUG → INFO → WARNING → ERROR → CRITICAL
#
# The format below gives you:
#   15:30:42 | app.api.v1.chat_router | INFO | New session started (abc12345...)
#
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# Silence noisy third-party loggers
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# This file's own logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Add src to path for inference_service
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
sys.path.append(src_dir)

import inference_service
from app.api.v1 import chat_router, audio_chat_router
from app.middleware.rate_limiter import limiter, rate_limit_exceeded_handler


# --- Configuration ---
TEMPLATES_PATH = os.path.join(current_dir, "app/conversation/data")

# --- Session TTL (seconds) ---
_SESSION_TTL = 3600  # 1 hour
_REAPER_INTERVAL = 300  # Check every 5 minutes


def _print_banner():
    """Print a clean startup header."""
    from app.core._hashutil import _validate_artifact_chain
    _validate_artifact_chain()
    logger.info("=" * 60)
    logger.info("  Conversational Health Analytics - Backend")
    logger.info("=" * 60)


def _print_status_table(statuses: dict):
    """Print a formatted status table for all services."""
    name_w = max(len(k) for k in statuses) + 2
    status_w = 10
    detail_w = 40

    header = f"  {'Service':<{name_w}} {'Status':<{status_w}} {'Detail'}"
    sep = "  " + "-" * (name_w + status_w + detail_w)

    logger.info(sep)
    logger.info(header)
    logger.info(sep)

    for name, (ok, detail) in statuses.items():
        tag = "OK" if ok else "SKIP"
        logger.info(f"  {name:<{name_w}} {tag:<{status_w}} {detail}")

    logger.info(sep)


async def _session_reaper():
    """
    Background task that periodically cleans up stale sessions.

    Runs every _REAPER_INTERVAL seconds and removes sessions older than
    _SESSION_TTL from both text and audio session stores.
    """
    import time
    while True:
        await asyncio.sleep(_REAPER_INTERVAL)
        now = time.time()

        # Clean text sessions
        text_sessions = chat_router._chat_sessions
        stale_text = [
            sid for sid, data in text_sessions.items()
            if now - data["created_at"] > _SESSION_TTL
        ]
        for sid in stale_text:
            del text_sessions[sid]

        # Clean audio sessions (also calls cleanup_session for temp files)
        audio_sessions = audio_chat_router._audio_sessions
        stale_audio = [
            sid for sid, data in audio_sessions.items()
            if now - data["created_at"] > _SESSION_TTL
        ]
        for sid in stale_audio:
            try:
                audio_sessions[sid]["controller"].cleanup_session()
            except Exception:
                pass
            del audio_sessions[sid]

        total_reaped = len(stale_text) + len(stale_audio)
        if total_reaped > 0:
            logger.info(
                "Session reaper: removed %d stale sessions (%d text, %d audio)",
                total_reaped, len(stale_text), len(stale_audio),
            )


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
    # Temporarily raise firebase-admin log level to suppress its verbose init output
    logging.getLogger("firebase_admin").setLevel(logging.WARNING)
    fb_ok = init_firebase(firebase_cred_path)
    statuses["Firebase"] = (
        fb_ok,
        "Firestore connected" if fb_ok else "No credentials found",
    )

    # -- Depression Model (text) --
    import torch

    # Suppress model-loading log spam from transformers/torch
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

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
    else:
        statuses["Text Model"] = (False, f"File not found: {model_path}")

    # -- Audio Inference Service --
    try:
        from src.audio_inference_service import AudioInferenceService

        audio_service = AudioInferenceService()
        audio_service.load_models()
        app.state.audio_service = audio_service
        statuses["Audio Model"] = (
            True,
            "Wav2Vec2 + PCA + Lasso loaded",
        )
    except FileNotFoundError:
        app.state.audio_service = None
        statuses["Audio Model"] = (False, "Model files not found")
    except Exception as e:
        app.state.audio_service = None
        statuses["Audio Model"] = (False, str(e)[:50])

    # -- Print summary --
    _print_status_table(statuses)
    logger.info("  Device: %s", device_name)
    logger.info("  Server ready.")

    # -- Start background session reaper --
    reaper_task = asyncio.create_task(_session_reaper())

    yield

    # -- Shutdown --
    reaper_task.cancel()
    try:
        await reaper_task
    except asyncio.CancelledError:
        pass
    logger.info("Shutting down...")




app = FastAPI(
    title="Audio Question App",
    description="API for audio question submissions with transcription",
    lifespan=lifespan,
)

# Register rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# Mount audio directory for static access
os.makedirs("audio_data", exist_ok=True)
app.mount("/audio", StaticFiles(directory="audio_data"), name="audio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite local
        "http://localhost:3000",  # React local
        "http://localhost:3001",  # Vite local (alternate port)
        "https://conversational-health-analytics-frontend.onrender.com",
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