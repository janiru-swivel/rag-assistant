from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.models import ConversationCreate, ConversationResponse, ConversationDetail
from app.database import get_db
from app.checkpointer import ChatCheckpointer
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/history", tags=["history"])

@router.post("/conversations", response_model=ConversationResponse, status_code=201)
async def create_conversation(
    request: ConversationCreate,
    db: Session = Depends(get_db)
):
    """Create a new conversation."""
    try:
        checkpointer = ChatCheckpointer(db)
        conversation = checkpointer.create_conversation(request.title)
        
        return ConversationResponse(
            id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at.isoformat(),
            updated_at=conversation.updated_at.isoformat(),
            message_count=0
        )
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating conversation: {str(e)}")

@router.get("/conversations", response_model=List[ConversationResponse])
async def get_all_conversations(db: Session = Depends(get_db)):
    """Get all conversations with message counts."""
    try:
        checkpointer = ChatCheckpointer(db)
        conversations = checkpointer.get_all_conversations()
        
        result = []
        for conv in conversations:
            message_count = len(conv.messages) if conv.messages else 0
            result.append(ConversationResponse(
                id=conv.id,
                title=conv.title,
                created_at=conv.created_at.isoformat(),
                updated_at=conv.updated_at.isoformat(),
                message_count=message_count
            ))
        
        return result
    except Exception as e:
        logger.error(f"Error getting conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting conversations: {str(e)}")

@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation_detail(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """Get details of a specific conversation including all messages."""
    try:
        checkpointer = ChatCheckpointer(db)
        conversation = checkpointer.get_conversation(conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get conversation state with messages
        state = checkpointer.restore_conversation_state(conversation_id)
        if not state:
            raise HTTPException(status_code=500, detail="Failed to restore conversation state")
        
        return ConversationDetail(
            id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at.isoformat(),
            updated_at=conversation.updated_at.isoformat(),
            messages=state["messages"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation {conversation_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting conversation: {str(e)}")

@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """Delete a conversation and all its messages."""
    try:
        checkpointer = ChatCheckpointer(db)
        success = checkpointer.delete_conversation(conversation_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return None
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation {conversation_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting conversation: {str(e)}")

@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """Get all messages for a specific conversation."""
    try:
        checkpointer = ChatCheckpointer(db)
        conversation = checkpointer.get_conversation(conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        messages = checkpointer.get_conversation_history(conversation_id)
        return messages
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting messages for conversation {conversation_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting messages: {str(e)}")

@router.get("/conversations/{conversation_id}/checkpoint")
async def get_conversation_checkpoint(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """Get the latest checkpoint for a conversation."""
    try:
        checkpointer = ChatCheckpointer(db)
        checkpoint = checkpointer.get_latest_checkpoint(conversation_id)
        
        if not checkpoint:
            raise HTTPException(status_code=404, detail="No checkpoint found for conversation")
        
        return checkpoint
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting checkpoint for conversation {conversation_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting checkpoint: {str(e)}")

@router.post("/conversations/{conversation_id}/restore")
async def restore_conversation(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """Restore conversation state from checkpoint."""
    try:
        checkpointer = ChatCheckpointer(db)
        state = checkpointer.restore_conversation_state(conversation_id)
        
        if not state:
            raise HTTPException(status_code=404, detail="Cannot restore conversation state")
        
        return {
            "message": "Conversation state restored successfully",
            "state": state
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restoring conversation {conversation_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error restoring conversation: {str(e)}")