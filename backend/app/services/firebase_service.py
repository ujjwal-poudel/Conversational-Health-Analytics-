"""
Firebase Firestore Service

Lightweight wrapper around firebase-admin for persisting conversation data.
Gracefully no-ops if Firebase is not configured (e.g. local dev without creds).
"""

import os
import logging
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)

# Module-level state
_firestore_db = None
_initialized = False


def init_firebase(cred_path: Optional[str] = None) -> bool:
    """
    Initialize Firebase Admin SDK with a service account JSON file.
    
    Parameters
    ----------
    cred_path : str, optional
        Path to the Firebase service account JSON file.
        Falls back to FIREBASE_CREDENTIALS_PATH env var.
    
    Returns
    -------
    bool
        True if initialization succeeded, False otherwise.
    """
    global _firestore_db, _initialized

    if _initialized:
        return _firestore_db is not None

    _initialized = True

    # Resolve credentials path
    path = cred_path or os.getenv("FIREBASE_CREDENTIALS_PATH")
    if not path:
        logger.warning("No credentials path provided. Firestore disabled.")
        return False

    if not os.path.exists(path):
        logger.warning("Credentials file not found: %s. Firestore disabled.", path)
        return False

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        cred = credentials.Certificate(path)
        firebase_admin.initialize_app(cred)
        _firestore_db = firestore.client()
        logger.info("Firestore initialized successfully.")
        return True
    except Exception as e:
        logger.error("Failed to initialize: %s", e)
        return False


def save_conversation(
    mode: str,
    turns: List[str],
    depression_score: Optional[float] = None,
    score_source: Optional[str] = None,
) -> Optional[str]:
    """
    Save a completed conversation to Firestore.
    
    Parameters
    ----------
    mode : str
        "text" or "audio"
    turns : list of str
        Conversation turns [Q, A, Q, A, ...]
    depression_score : float, optional
        Final depression score
    score_source : str, optional
        "text", "fusion", or "audio"
    
    Returns
    -------
    str or None
        The Firestore document ID if saved, None if skipped/failed.
    """
    if _firestore_db is None:
        logger.info("Firestore not initialized. Skipping save.")
        return None

    try:
        doc_data = {
            "mode": mode,
            "turns": turns,
            "depression_score": depression_score,
            "score_source": score_source,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        doc_ref = _firestore_db.collection("conversations").add(doc_data)
        doc_id = doc_ref[1].id
        logger.info("Saved conversation to Firestore (doc: %s)", doc_id)
        return doc_id

    except Exception as e:
        logger.error("Failed to save conversation: %s", e)
        return None
