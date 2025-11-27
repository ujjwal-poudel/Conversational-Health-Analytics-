from pydantic import BaseModel
from typing import Optional, List, Union

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: Union[str, List[str]]  # Support both single string and list of message parts
    is_finished: bool
    depression_score: Optional[float] = None
    semantic_risk_label: Optional[str] = None
    consistency_status: Optional[str] = None