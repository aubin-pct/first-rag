from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

# Modèle de données pour la requête
class ChatRequest(BaseModel):
    question: str
    history: List[dict] = [] # Liste de {"role": "user"|"assistant", "content": "..."}
    session_id: Optional[int] = None

# Modèle de données pour la réponse
class ChatResponse(BaseModel):
    answer: str
    context: List[str] = []
    session_id: int

# Modèle pour une session de chat (liste)
class ChatSessionSchema(BaseModel):
    id: int
    title: Optional[str]
    created_at: datetime
    is_pinned: bool = False

    class Config:
        from_attributes = True

# Modèle pour un message de chat
class ChatMessageSchema(BaseModel):
    id: int
    role: str
    content: str
    timestamp: datetime

    class Config:
        from_attributes = True
