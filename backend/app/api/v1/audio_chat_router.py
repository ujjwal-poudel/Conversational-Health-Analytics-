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
# TODO: Add session cleanup and persistence
_audio_sessions = {}


def _convert_path_to_url(path: str) -> str:
    """Convert local file path to static URL."""
    if not path:
        return ""
    # Normalize path separators
    path = path.replace("\\", "/")
    
    if "audio_data/" in path:
        relative_path = path.split("audio_data/")[-1]
        # Return relative URL path (frontend prepends base URL)
        return f"/audio/{relative_path}"
    return path


@router.post("/start")
async def start_audio_chat(request: Request, session_id: str = Form(...)):
    """
    Start a new audio conversation.
    
    Parameters
    ----------
    session_id : str (form field)
        Unique identifier for this audio conversation session.

    Returns
    -------
    JSONResponse
        {
            "response_text": list of str,
            "response_audio_paths": list of str,
            "is_finished": bool
        }
    """
    print(f"\n[AUDIO] Starting new conversation for session: {session_id}")
    
    templates_path = getattr(request.app.state, "templates_path", None)
    
    if not templates_path:
        print("[AUDIO] ERROR: Templates path not found in app state!")
        raise HTTPException(status_code=503, detail="Templates not initialized")
    
    try:
        controller = AudioConversationController(
            session_id=session_id,
            templates_base_path=templates_path
        )
        
        # Store controller for this session
        _audio_sessions[session_id] = controller
        
        result = controller.start_conversation()
        print(f"[AUDIO] First message parts: {result['response_text']}")
        
        # Convert audio paths to URLs
        if 'response_audio_paths' in result:
            result['response_audio_paths'] = [_convert_path_to_url(p) for p in result['response_audio_paths']]
        
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"[AUDIO] Error starting conversation: {e}")
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
    """
    Process one conversational turn in audio mode.

    Parameters
    ----------
    session_id : str (form field)
        Unique identifier used to maintain conversation context.

    audio_file : UploadFile (multipart/form-data)
        The WAV audio file containing the user's spoken response.

    Returns
    -------
    JSONResponse
        {
            "transcript": str,
            "user_audio_path": str,
            "response_text": list of str,
            "response_audio_paths": list of str,
            "is_finished": bool
        }

    Raises
    ------
    HTTPException
        If audio is missing, invalid, or processing fails at any stage.
    """
    print(f"\n[AUDIO] Processing turn for session: {session_id}")
    
    # Get existing controller for this session
    controller = _audio_sessions.get(session_id)
    
    if not controller:
        raise HTTPException(
            status_code=400,
            detail=f"No active conversation for session {session_id}. Please call /start first."
        )

    # Validate file input
    if audio_file is None:
        raise HTTPException(
            status_code=400,
            detail="Missing audio file input."
        )

    try:
        wav_bytes = await audio_file.read()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read uploaded audio: {str(e)}"
        )

    try:
        result = controller.process_audio_turn(wav_bytes)
        print(f"[AUDIO] Transcript: {result['transcript']}")
        print(f"[AUDIO] Bot response parts: {result['response_text']}")
        
        # If conversation is finished, calculate score and save everything
        if result.get('is_finished', False):
            print("[AUDIO] Conversation finished. Processing final steps...")
            
            try:
                # Get transcript for regression model
                transcript = controller.get_inference_pairs()
                
                # Calculate depression score
                depression_model = getattr(request.app.state, "depression_model", None)
                tokenizer = getattr(request.app.state, "depression_tokenizer", None)
                device = getattr(request.app.state, "device", None)
                
                depression_score = None
                semantic_risk_label = None
                consistency_status = None
                
                if depression_model and tokenizer and device:
                    import inference_service
                    from app.conversation.analysis.guardrails import analyze_with_semantic_guardrails
                    
                    # Get raw regression score
                    raw_score = inference_service.get_depression_score(
                        transcript_turns=transcript,
                        model=depression_model,
                        tokenizer=tokenizer,
                        device=device
                    )
                    
                    # Apply semantic guardrails
                    final_score, label, status, sim_0, sim_1 = analyze_with_semantic_guardrails(transcript, raw_score)
                    
                    depression_score = final_score
                    semantic_risk_label = label
                    consistency_status = status
                    
                    print(f"[AUDIO] Final Score: {depression_score}")
                    print(f"[AUDIO] Semantic Label: {semantic_risk_label}")
                    print(f"[AUDIO] Status: {consistency_status}")
                    
                    # Save to JSONL
                    try:
                        from app.routers.answer_router import get_next_id, save_to_jsonl
                        current_id = get_next_id()
                        save_to_jsonl(current_id, transcript, final_score)
                        print(f"[AUDIO] Saved conversation to user_data.jsonl with ID: {current_id}")
                        
                        # Merge and save audio files
                        try:
                            merged_audio_paths = controller.finalize_conversation(current_id)
                            print(f"[AUDIO] User-only audio: {merged_audio_paths['user_only_path']}")
                            print(f"[AUDIO] Full conversation audio: {merged_audio_paths['full_conversation_path']}")
                            
                            # Add merged paths to result
                            result['merged_audio'] = merged_audio_paths
                            
                        except Exception as e:
                            print(f"[AUDIO] WARNING: Failed to merge audio files: {e}")
                        
                    except Exception as e:
                        print(f"[AUDIO] WARNING: Failed to save user data: {e}")
                    
                    # Add scores to result
                    result['depression_score'] = depression_score
                    result['semantic_risk_label'] = semantic_risk_label
                    result['consistency_status'] = consistency_status
                    
                else:
                    print("[AUDIO] WARNING: Depression model not loaded. Skipping scoring.")
                    
            except Exception as e:
                print(f"[AUDIO] Error in final processing: {e}")
        
        # Convert audio paths to URLs
        if 'response_audio_paths' in result:
            result['response_audio_paths'] = [_convert_path_to_url(p) for p in result['response_audio_paths']]

        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"[AUDIO] Error processing turn: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Audio conversation processing failed: {str(e)}"
        )


@router.post("/cleanup")
async def cleanup_session(session_id: str = Form(...)):
    """
    Clean up session audio files.
    Should be called by frontend after final playback is complete.
    """
    print(f"\n[AUDIO] Cleaning up session: {session_id}")
    
    controller = _audio_sessions.get(session_id)
    if not controller:
        # Session might already be cleaned up or invalid
        return JSONResponse(content={"status": "skipped", "reason": "Session not found"})
        
    try:
        controller.cleanup_session()
        # Remove from active sessions
        del _audio_sessions[session_id]
        return JSONResponse(content={"status": "success"})
    except Exception as e:
        print(f"[AUDIO] Error cleaning up session: {e}")
        raise HTTPException(status_code=500, detail=str(e))