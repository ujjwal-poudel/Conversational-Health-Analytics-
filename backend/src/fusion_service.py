"""
Fusion Service for Multimodal Depression Score Prediction

Combines text and audio model predictions using min fusion strategy.
Provides parallel execution with timeout and graceful fallback.

Fusion Strategy:
- Both succeed: min(text_score, audio_score)
- One fails: return the successful one
- Both fail: return None with error

Features:
- Parallel execution via asyncio
- 40-second timeout per model
- Graceful fallback logic
- Comprehensive error logging
"""

import asyncio
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum


class ScoreSource(Enum):
    """Source of the final depression score."""
    FUSION = "fusion"      # Both models succeeded, min fusion applied
    TEXT_ONLY = "text"     # Only text model succeeded
    AUDIO_ONLY = "audio"   # Only audio model succeeded
    FAILED = "failed"      # Both models failed


@dataclass
class FusionResult:
    """Result of multimodal fusion."""
    score: Optional[float]
    source: ScoreSource
    text_score: Optional[float]
    audio_score: Optional[float]
    text_error: Optional[str]
    audio_error: Optional[str]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "score": self.score,
            "source": self.source.value,
            "text_score": self.text_score,
            "audio_score": self.audio_score,
            "text_error": self.text_error,
            "audio_error": self.audio_error,
        }


async def _run_text_inference(
    transcript_turns: List[str],
    model,
    tokenizer,
    device,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Run text model inference asynchronously.
    
    Returns (score, error_message).
    """
    try:
        # Import here to avoid circular imports
        import inference_service
        
        # Run in thread pool since it's CPU/GPU bound
        loop = asyncio.get_event_loop()
        score = await loop.run_in_executor(
            None,
            lambda: inference_service.get_depression_score(
                transcript_turns=transcript_turns,
                model=model,
                tokenizer=tokenizer,
                device=device,
                turn_batch_size=16
            )
        )
        return score, None
    
    except Exception as e:
        error_msg = f"Text inference failed: {str(e)}"
        print(f"[FUSION] {error_msg}")
        return None, error_msg


async def _run_audio_inference(
    audio_path: str,
    audio_service,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Run audio model inference asynchronously.
    
    Returns (score, error_message).
    """
    if audio_service is None:
        return None, "Audio service not available"
    
    if not audio_service.is_loaded():
        return None, "Audio models not loaded"
    
    try:
        # Run in thread pool since it's CPU/GPU bound
        loop = asyncio.get_event_loop()
        score = await loop.run_in_executor(
            None,
            lambda: audio_service.predict(audio_path)
        )
        
        if score is None:
            return None, "Audio prediction returned None"
        
        return score, None
    
    except Exception as e:
        error_msg = f"Audio inference failed: {str(e)}"
        print(f"[FUSION] {error_msg}")
        return None, error_msg


async def get_fused_score(
    transcript_turns: List[str],
    audio_path: str,
    text_model,
    tokenizer,
    device,
    audio_service,
    timeout: float = 40.0,
) -> FusionResult:
    """
    Run both text and audio models in parallel with min fusion.
    
    Parameters
    ----------
    transcript_turns : list of str
        Conversation turns for text model.
    audio_path : str
        Path to merged user audio file.
    text_model : PHQTotalMulticlassAttentionModelBERT
        Loaded text model.
    tokenizer : AutoTokenizer
        Loaded tokenizer.
    device : torch.device
        Compute device.
    audio_service : AudioInferenceService
        Loaded audio inference service.
    timeout : float
        Timeout in seconds (default: 40.0).
    
    Returns
    -------
    FusionResult
        Contains final score, source, individual scores, and errors.
    """
    print(f"[FUSION] Starting parallel inference (timeout: {timeout}s)")
    
    # Create tasks for parallel execution
    text_task = asyncio.create_task(
        _run_text_inference(transcript_turns, text_model, tokenizer, device)
    )
    audio_task = asyncio.create_task(
        _run_audio_inference(audio_path, audio_service)
    )
    
    # Initialize results
    text_score = None
    audio_score = None
    text_error = None
    audio_error = None
    
    try:
        # Wait for both with timeout
        results = await asyncio.wait_for(
            asyncio.gather(text_task, audio_task, return_exceptions=True),
            timeout=timeout
        )
        
        # Process text result
        if isinstance(results[0], Exception):
            text_error = f"Text task exception: {str(results[0])}"
        else:
            text_score, text_error = results[0]
        
        # Process audio result
        if isinstance(results[1], Exception):
            audio_error = f"Audio task exception: {str(results[1])}"
        else:
            audio_score, audio_error = results[1]
    
    except asyncio.TimeoutError:
        print(f"[FUSION] Timeout after {timeout}s - checking partial results")
        
        # Check if text completed
        if text_task.done() and not text_task.cancelled():
            try:
                result = text_task.result()
                if not isinstance(result, Exception):
                    text_score, text_error = result
            except Exception as e:
                text_error = f"Text task failed: {str(e)}"
        else:
            text_error = "Text inference timed out"
            text_task.cancel()
        
        # Check if audio completed
        if audio_task.done() and not audio_task.cancelled():
            try:
                result = audio_task.result()
                if not isinstance(result, Exception):
                    audio_score, audio_error = result
            except Exception as e:
                audio_error = f"Audio task failed: {str(e)}"
        else:
            audio_error = "Audio inference timed out"
            audio_task.cancel()
    
    # Determine final score and source
    final_score = None
    source = ScoreSource.FAILED
    
    if text_score is not None and audio_score is not None:
        # Both succeeded
        if text_score == 0 or audio_score == 0:
            final_score = (text_score + audio_score) / 2
            source = ScoreSource.FUSION
            print(f"[FUSION] Zero detected - using average: ({text_score:.4f} + {audio_score:.4f}) / 2 = {final_score:.4f}")
        else:
            final_score = min(text_score, audio_score)
            source = ScoreSource.FUSION
            print(f"[FUSION] Standard min fusion: min({text_score:.4f}, {audio_score:.4f}) = {final_score:.4f}")
    
    elif text_score is not None:
        # Only text succeeded
        final_score = text_score
        source = ScoreSource.TEXT_ONLY
        print(f"[FUSION] Fallback to text-only: {final_score:.4f}")
    
    elif audio_score is not None:
        # Only audio succeeded
        final_score = audio_score
        source = ScoreSource.AUDIO_ONLY
        print(f"[FUSION] Fallback to audio-only: {final_score:.4f}")
    
    else:
        # Both failed
        print("[FUSION] Both models failed - no score available")
    
    return FusionResult(
        score=final_score,
        source=source,
        text_score=text_score,
        audio_score=audio_score,
        text_error=text_error,
        audio_error=audio_error,
    )


def get_fused_score_sync(
    transcript_turns: List[str],
    audio_path: str,
    text_model,
    tokenizer,
    device,
    audio_service,
    timeout: float = 40.0,
) -> FusionResult:
    """
    Synchronous wrapper for get_fused_score.
    Use this when calling from a sync context.
    """
    return asyncio.run(
        get_fused_score(
            transcript_turns=transcript_turns,
            audio_path=audio_path,
            text_model=text_model,
            tokenizer=tokenizer,
            device=device,
            audio_service=audio_service,
            timeout=timeout,
        )
    )
