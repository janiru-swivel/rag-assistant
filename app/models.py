from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[int] = None

class QueryResponse(BaseModel):
    id: int
    query: str
    answer: str
    sources: List[str]
    created_at: str
    conversation_id: int
    references: List[str] = []  # For compatibility with Postman collection

    def __init__(self, **data):
        # Ensure references is set to sources for backward compatibility
        if 'sources' in data and 'references' not in data:
            data['references'] = data['sources']
        super().__init__(**data)

class ConversationCreate(BaseModel):
    title: str

class ConversationResponse(BaseModel):
    id: int
    title: str
    created_at: str
    updated_at: str
    message_count: int

class ConversationDetail(BaseModel):
    id: int
    title: str
    created_at: str
    updated_at: str
    messages: List[dict]

class MessageResponse(BaseModel):
    id: int
    query: str
    answer: str
    sources: List[str]
    created_at: str