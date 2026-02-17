import uuid
import time
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.schemas.chat import ChatRequest, ChatResponse
from app.conversation.engine import ConversationEngine
import inference_service

router = APIRouter()

# ── Session storage ──────────────────────────────────────────────────
# Mirrors the pattern used by audio_chat_router._audio_sessions

_chat_sessions: dict[str, dict] = {}
# Each value: {"engine": ConversationEngine, "created_at": float}

_SESSION_TTL_SECONDS = 3600  # 1 hour


def _cleanup_stale_sessions() -> None:
    """Remove sessions older than _SESSION_TTL_SECONDS."""
    now = time.time()
    stale = [
        sid for sid, data in _chat_sessions.items()
        if now - data["created_at"] > _SESSION_TTL_SECONDS
    ]
    for sid in stale:
        del _chat_sessions[sid]


def _get_engine(session_id: Optional[str]) -> ConversationEngine:
    """Look up a ConversationEngine by session_id, or raise 400."""
    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="Missing session_id. Call /start first to obtain one.",
        )
    session = _chat_sessions.get(session_id)
    if not session:
        raise HTTPException(
            status_code=400,
            detail=f"No active conversation for session {session_id}. Call /start first.",
        )
    return session["engine"]


# ── Routes ───────────────────────────────────────────────────────────

@router.post("/start", response_model=ChatResponse)
async def start_chat(request: Request):
    _cleanup_stale_sessions()

    templates_path = getattr(request.app.state, "templates_path", None)
    if not templates_path:
        raise HTTPException(status_code=503, detail="Templates not initialized")

    session_id = str(uuid.uuid4())

    new_engine = ConversationEngine(templates_base_path=templates_path)
    _chat_sessions[session_id] = {
        "engine": new_engine,
        "created_at": time.time(),
    }

    first_message = new_engine.start()

    if isinstance(first_message, str):
        first_message = [first_message]

    print(f"\n  [CHAT] New session started ({session_id[:8]}...)")

    return ChatResponse(
        response=first_message,
        is_finished=False,
        depression_score=None,
        session_id=session_id,
    )


@router.post("/message", response_model=ChatResponse)
async def chat_message(request: Request, chat_request: ChatRequest):
    engine = _get_engine(chat_request.session_id)

    user_message = chat_request.message
    bot_response = engine.process(user_message)

    if isinstance(bot_response, str):
        bot_response = [bot_response]

    is_finished = engine.is_finished()
    depression_score = None

    if is_finished:
        depression_model = getattr(request.app.state, "depression_model", None)
        tokenizer = getattr(request.app.state, "depression_tokenizer", None)
        device = getattr(request.app.state, "device", None)

        if depression_model and tokenizer and device:
            transcript = engine.get_inference_pairs()
            try:
                depression_score = inference_service.get_depression_score(
                    transcript_turns=transcript,
                    model=depression_model,
                    tokenizer=tokenizer,
                    device=device,
                )
                print(f"  [CHAT] Conversation complete  |  Score: {depression_score:.4f}")

                # Save to Firestore
                try:
                    from app.services.firebase_service import save_conversation
                    save_conversation(
                        mode="text",
                        turns=transcript,
                        depression_score=depression_score,
                        score_source="text",
                    )
                except Exception as e:
                    print(f"  [CHAT] WARNING: Firestore save failed: {e}")

            except Exception as e:
                print(f"  [CHAT] ERROR: Scoring failed: {e}")
        else:
            print("  [CHAT] WARNING: Depression model not loaded. Skipping scoring.")

    return ChatResponse(
        response=bot_response,
        is_finished=is_finished,
        depression_score=depression_score,
        session_id=chat_request.session_id,
    )


@router.post("/cleanup")
async def cleanup_chat(request: Request):
    """
    Clean up/reset a text conversation session.
    Parses body manually to handle sendBeacon's text/plain content-type.
    """
    session_id = None
    try:
        body = await request.json()
        session_id = body.get("session_id")
    except Exception:
        pass  # sendBeacon might send malformed or empty body

    if session_id and session_id in _chat_sessions:
        del _chat_sessions[session_id]
        print(f"  [CHAT] Session {session_id[:8]}... cleaned up.")
        return {"status": "success"}

    return {"status": "skipped", "reason": "No session_id provided or session not found"}