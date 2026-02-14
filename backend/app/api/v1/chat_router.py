from fastapi import APIRouter, Request, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.conversation.engine import ConversationEngine
import inference_service

router = APIRouter()

@router.post("/message", response_model=ChatResponse)
async def chat_message(request: Request, chat_request: ChatRequest):
    print(f"\n[CHAT] Received message: '{chat_request.message}'")
    engine = getattr(request.app.state, "conversation_engine", None)
    if not engine:
        print("[CHAT] ERROR: Conversation engine not initialized!")
        raise HTTPException(status_code=503, detail="Conversation engine not initialized")

    user_message = chat_request.message
    
    # Process message - engine now returns list of message parts
    bot_response = engine.process(user_message)
    
    # Ensure response is always a list
    if isinstance(bot_response, str):
        bot_response = [bot_response]
    
    print(f"[CHAT] Bot response parts: {bot_response}")
    
    is_finished = engine.is_finished()
    
    depression_score = None
    
    if is_finished:
        print("[CHAT] Conversation finished. Calculating score...")
        # Calculate depression score
        depression_model = getattr(request.app.state, "depression_model", None)
        tokenizer = getattr(request.app.state, "depression_tokenizer", None)
        device = getattr(request.app.state, "device", None)
        
        if depression_model and tokenizer and device:
            transcript = engine.get_inference_pairs()
            try:
                # Get depression score
                depression_score = inference_service.get_depression_score(
                    transcript_turns=transcript,
                    model=depression_model,
                    tokenizer=tokenizer,
                    device=device
                )
                
                print(f"[CHAT] Depression Score: {depression_score:.4f}")
                
                # Save to JSONL
                try:
                    from app.routers.answer_router import get_next_id, save_to_jsonl
                    current_id = get_next_id()
                    save_to_jsonl(current_id, transcript, depression_score)
                    print(f"[CHAT] Saved conversation to user_data.jsonl with ID: {current_id}")
                except Exception as e:
                    print(f"[CHAT] WARNING: Failed to save user data: {e}")
                
            except Exception as e:
                print(f"[CHAT] Error calculating depression score: {e}")
        else:
            print("[CHAT] WARNING: Depression model not loaded. Skipping scoring.")
    
    return ChatResponse(
        response=bot_response,
        is_finished=is_finished,
        depression_score=depression_score
    )

@router.post("/start", response_model=ChatResponse)
async def start_chat(request: Request):
    print("\n[CHAT] Starting new conversation...")
    # Reset the engine by creating a new one
    templates_path = getattr(request.app.state, "templates_path", None)
    
    if not templates_path:
        print("[CHAT] ERROR: Templates path not found in app state!")
        raise HTTPException(status_code=503, detail="Templates not initialized")
        
    new_engine = ConversationEngine(
        templates_base_path=templates_path
    )
    
    # Update the global state (Single User Mode)
    request.app.state.conversation_engine = new_engine
    
    # Start the conversation
    first_message = new_engine.start()
    
    # Ensure response is always a list
    if isinstance(first_message, str):
        first_message = [first_message]
    
    print(f"[CHAT] First question parts: {first_message}")
    
    return ChatResponse(
        response=first_message,  # Now returns list
        is_finished=False,
        depression_score=None
    )


@router.post("/cleanup")
async def cleanup_chat(request: Request):
    """
    Clean up/reset the text conversation session.
    """
    print("\n[CHAT] Cleanup requested. Resetting conversation engine.")
    # Reset the global engine to clear in-memory state
    request.app.state.conversation_engine = None
    return {"status": "success"}