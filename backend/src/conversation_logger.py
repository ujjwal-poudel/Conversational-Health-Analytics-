"""
Conversation Logger

Logs completed conversations with their final scores to a text file
on the external harddrive for record-keeping and analysis.

Only complete, successful conversations are logged (not cancelled ones).
"""

import os
from datetime import datetime
from typing import List, Optional


# Configuration
LOG_DIR = "/Volumes/MACBACKUP/logs"
LOG_FILE = os.path.join(LOG_DIR, "conversation_logs.txt")


def _ensure_log_dir():
    """Create log directory if it doesn't exist."""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        return True
    except Exception as e:
        print(f"[LOGGER] Cannot create log directory: {e}")
        return False


def log_completed_conversation(
    conversation_id: str,
    turns: List[str],
    final_score: float,
    source: str,
    semantic_label: Optional[str] = None,
) -> bool:
    """
    Log a completed conversation to the external harddrive.
    
    Parameters
    ----------
    conversation_id : str
        Unique identifier for this conversation.
    turns : list of str
        List of conversation turns [Q, A, Q, A, ...].
    final_score : float
        Final depression score after guardrails.
    source : str
        Score source: "fusion", "text", or "audio".
    semantic_label : str, optional
        Semantic risk label from guardrails.
    
    Returns
    -------
    bool
        True if logging succeeded, False otherwise.
    """
    if not _ensure_log_dir():
        return False
    
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format conversation turns
        formatted_turns = []
        for i, turn in enumerate(turns):
            role = "Q" if i % 2 == 0 else "A"
            formatted_turns.append(f"    {role}: {turn}")
        turns_text = "\n".join(formatted_turns)
        
        # Build log entry
        log_entry = f"""
{'=' * 70}
CONVERSATION LOG
{'=' * 70}
ID:            {conversation_id}
Timestamp:     {timestamp}
Score Source:  {source.upper()}
Final Score:   {final_score:.2f}
Semantic:      {semantic_label or 'N/A'}
{'-' * 70}
TURNS:
{turns_text}
{'=' * 70}

"""
        
        # Append to log file
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        print(f"[LOGGER] Logged conversation {conversation_id} to {LOG_FILE}")
        return True
    
    except Exception as e:
        print(f"[LOGGER] Failed to log conversation: {e}")
        return False


def get_log_file_path() -> str:
    """Return the path to the log file."""
    return LOG_FILE
