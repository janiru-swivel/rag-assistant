from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from datetime import datetime

router = APIRouter(prefix="/history", tags=["history"])

class ChatHistory(BaseModel):
    query: str
    answer: str
    timestamp: str

# In-memory storage for demo purposes
chat_history: List[ChatHistory] = []

@router.get("/", response_model=List[ChatHistory])
async def get_chat_history():
    """Retrieve the chat history."""
    return chat_history

@router.post("/")
async def add_chat_history(history: ChatHistory):
    """Add a new entry to the chat history."""
    history.timestamp = datetime.utcnow().isoformat()
    chat_history.append(history)
    return {"message": "History added successfully"}