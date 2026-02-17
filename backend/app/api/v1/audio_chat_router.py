"""
audio_chat_router.py

This module defines the FastAPI routes for the audio-based conversation system.
It exposes endpoints for audio-based conversations:

- /start: Initialize a new audio conversation
- /turn: Process user audio and get bot audio response

This router is intentionally thin and delegates all business logic to the
AudioConversationController, ensuring a clean separation between:

    - API layer
    - Controller layer
    - Orchestration layer
    - STT/TTS services
    - Core conversation engine
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse

from app.conversation.controller.audio_conversation_controller import (
    AudioConversationController,
)

router = APIRouter(
    prefix="/audio/chat",
    tags=["Audio Conversation"],
)

# Store audio conversation controllers per session
_audio_sessions = {}


def _convert_path_to_url(path: str) -> str:
    """Convert local file path to static URL."""
    if not path:
        return ""
    path = path.replace("\\", "/")

    if "audio_data/" in path:
        relative_path = path.split("audio_data/")[-1]
        return f"/audio/{relative_path}"
    return path


@router.post("/start")
async def start_audio_chat(request: Request, session_id: str = Form(...)):
    """Start a new audio conversation."""
    templates_path = getattr(request.app.state, "templates_path", None)

    if not templates_path:
        raise HTTPException(status_code=503, detail="Templates not initialized")

    try:
        controller = AudioConversationController(
            session_id=session_id,
            templates_base_path=templates_path
        )

        _audio_sessions[session_id] = controller

        result = await controller.start_conversation()

        print(f"\n  [AUDIO] New session started ({session_id})")

        if 'response_audio_paths' in result:
            result['response_audio_paths'] = [_convert_path_to_url(p) for p in result['response_audio_paths']]

        return JSONResponse(content=result)

    except Exception as e:
        print(f"  [AUDIO] ERROR: Failed to start session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start audio conversation: {str(e)}"
        )


@router.post("/turn")
async def process_audio_turn(
    request: Request,
    session_id: str = Form(...),
    audio_file: UploadFile = File(...)
):
    """Process one conversational turn in audio mode."""
    controller = _audio_sessions.get(session_id)

    if not controller:
        raise HTTPException(
            status_code=400,
            detail=f"No active conversation for session {session_id}. Please call /start first."
        )

    if audio_file is None:
        raise HTTPException(status_code=400, detail="Missing audio file input.")

    try:
        wav_bytes = await audio_file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded audio: {str(e)}")

    try:
        result = await controller.process_audio_turn(wav_bytes)

        if result.get('is_finished', False):
            try:
                transcript = controller.get_inference_pairs()

                depression_model = getattr(request.app.state, "depression_model", None)
                tokenizer = getattr(request.app.state, "depression_tokenizer", None)
                device = getattr(request.app.state, "device", None)
                audio_service = getattr(request.app.state, "audio_service", None)

                depression_score = None
                score_source = None

                if depression_model and tokenizer and device:

                    # Audio merge disabled — no longer saving audio files to disk
                    user_audio_path = None

                    if audio_service is not None and audio_service.is_loaded() and user_audio_path:
                        from src.fusion_service import get_fused_score
                        from fastapi.concurrency import run_in_threadpool

                        # Offload blocking Fusion inference
                        fusion_result = await run_in_threadpool(
                            get_fused_score,
                            transcript_turns=transcript,
                            audio_path=user_audio_path,
                            text_model=depression_model,
                            tokenizer=tokenizer,
                            device=device,
                            audio_service=audio_service,
                            timeout=40.0
                        )

                        raw_score = fusion_result.score
                        score_source = fusion_result.source.value

                        result['fusion_details'] = {
                            'text_score': fusion_result.text_score,
                            'audio_score': fusion_result.audio_score,
                            'source': score_source,
                        }
                    else:
                        # Fallback to text-only
                        import inference_service
                        from fastapi.concurrency import run_in_threadpool
                        
                        # Offload blocking Text inference
                        raw_score = await run_in_threadpool(
                            inference_service.get_depression_score,
                            transcript_turns=transcript,
                            model=depression_model,
                            tokenizer=tokenizer,
                            device=device
                        )
                        score_source = "text"

                    if raw_score is not None:
                        depression_score = raw_score

                        print(f"  [AUDIO] Conversation complete  |  Score: {depression_score:.4f} ({score_source})")

                        # Save to Firestore
                        try:
                            from app.services.firebase_service import save_conversation
                            save_conversation(
                                mode="audio",
                                turns=transcript,
                                depression_score=depression_score,
                                score_source=score_source,
                            )
                        except Exception as e:
                            print(f"  [AUDIO] WARNING: Firestore save failed: {e}")
                    else:
                        print("  [AUDIO] WARNING: No score available")

                    result['depression_score'] = depression_score
                    result['score_source'] = score_source

                else:
                    print("  [AUDIO] WARNING: Depression model not loaded. Skipping scoring.")

            except Exception as e:
                print(f"  [AUDIO] ERROR: Final processing failed: {e}")
                import traceback
                traceback.print_exc()

        # Convert audio paths to URLs
        if 'response_audio_paths' in result:
            result['response_audio_paths'] = [_convert_path_to_url(p) for p in result['response_audio_paths']]

        return JSONResponse(content=result)

    except Exception as e:
        print(f"  [AUDIO] ERROR: Turn processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Audio conversation processing failed: {str(e)}"
        )


@router.post("/cleanup")
async def cleanup_session(session_id: str = Form(...)):
    """Clean up session audio files."""
    controller = _audio_sessions.get(session_id)
    if not controller:
        return JSONResponse(content={"status": "skipped", "reason": "Session not found"})

    try:
        controller.cleanup_session()
        del _audio_sessions[session_id]
        print(f"  [AUDIO] Session {session_id} cleaned up.")
        return JSONResponse(content={"status": "success"})
    except Exception as e:
        print(f"  [AUDIO] ERROR: Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))