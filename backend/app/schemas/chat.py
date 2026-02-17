from pydantic import BaseModel
from typing import Optional, List, Union

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: Union[str, List[str]]  # Support both single string and list of message parts
    is_finished: bool
    depression_score: Optional[float] = None
    session_id: Optional[str] = None