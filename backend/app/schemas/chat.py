from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    is_finished: bool
    depression_score: Optional[float] = None
    semantic_risk_label: Optional[str] = None
    consistency_status: Optional[str] = None