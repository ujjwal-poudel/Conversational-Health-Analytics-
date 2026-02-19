import uuid
import time
import logging
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.schemas.chat import ChatRequest, ChatResponse
from app.conversation.engine import ConversationEngine
import inference_service
from app.middleware.rate_limiter import limiter

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Session storage ──────────────────────────────────────────────────
# Mirrors the pattern used by audio_chat_router._audio_sessions

_chat_sessions: dict[str, dict] = {}
# Each value: {"engine": ConversationEngine, "created_at": float}


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

from fastapi.concurrency import run_in_threadpool


@router.post("/start", response_model=ChatResponse)
@limiter.limit("10/minute")
async def start_chat(request: Request):

    templates_path = getattr(request.app.state, "templates_path", None)
    if not templates_path:
        raise HTTPException(status_code=503, detail="Templates not initialized")

    try:
        session_id = str(uuid.uuid4())

        new_engine = ConversationEngine(templates_base_path=templates_path)
        _chat_sessions[session_id] = {
            "engine": new_engine,
            "created_at": time.time(),
        }

        # Offload blocking LLM generation to threadpool
        first_message = await run_in_threadpool(new_engine.start)

        if isinstance(first_message, str):
            first_message = [first_message]

        logger.info("New session started (%s...)", session_id[:8])

        return ChatResponse(
            response=first_message,
            is_finished=False,
            depression_score=None,
            session_id=session_id,
        )
    except Exception as e:
        logger.error("Start failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to start conversation")


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(chat_request: ChatRequest, request: Request):
    engine = _get_engine(chat_request.session_id)

    user_message = chat_request.message
    
    # Offload blocking LLM generation to threadpool
    try:
        bot_response = await run_in_threadpool(engine.process, user_message)
    except Exception as e:
        logger.error("Engine process failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to generate response")

    if isinstance(bot_response, str):
        bot_response = [bot_response]

    is_finished = await run_in_threadpool(engine.is_finished)
    depression_score = None

    if is_finished:
        depression_model = getattr(request.app.state, "depression_model", None)
        tokenizer = getattr(request.app.state, "depression_tokenizer", None)
        device = getattr(request.app.state, "device", None)

        if depression_model and tokenizer and device:
            transcript = engine.get_inference_pairs()
            try:
                # Offload blocking inference to threadpool
                depression_score = await run_in_threadpool(
                    inference_service.get_depression_score,
                    transcript_turns=transcript,
                    model=depression_model,
                    tokenizer=tokenizer,
                    device=device,
                )
                logger.info("Conversation complete  |  Score: %.4f", depression_score)

                # Save to Firestore (IO bound but fast, acceptable in threadpool or async)
                try:
                    from app.services.firebase_service import save_conversation
                    # Ideally firebase calls should be async or threadpool too if blocking
                    await run_in_threadpool(
                        save_conversation,
                        mode="text",
                        turns=transcript,
                        depression_score=depression_score,
                        score_source="text",
                    )
                except Exception as e:
                    logger.warning("Firestore save failed: %s", e)

            except Exception as e:
                logger.error("Scoring failed: %s", e)
        else:
            logger.warning("Depression model not loaded. Skipping scoring.")

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
        logger.info("Session %s... cleaned up.", session_id[:8])
        return {"status": "success"}

    return {"status": "skipped", "reason": "No session_id provided or session not found"}