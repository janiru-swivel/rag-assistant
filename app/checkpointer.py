import json
import logging
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class ChatCheckpointer:
    """Manages conversation checkpoints for state persistence and recovery"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_conversation(self, title: str):
        """Create a new conversation"""
        try:
            from .database import Conversation
            conversation = Conversation(title=title)
            self.db.add(conversation)
            self.db.commit()
            self.db.refresh(conversation)
            
            # Create initial checkpoint
            self.save_checkpoint(conversation.id, {
                "conversation_id": conversation.id,
                "title": title,
                "message_count": 0,
                "created_at": conversation.created_at.isoformat(),
                "messages": []
            })
            
            logger.info(f"Created new conversation: {conversation.id}")
            return conversation
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to create conversation: {e}")
            raise
    
    def get_conversation(self, conversation_id: int):
        """Get conversation by ID"""
        try:
            from .database import Conversation
            return self.db.query(Conversation).filter(
                Conversation.id == conversation_id,
                Conversation.is_active == True
            ).first()
        except Exception as e:
            logger.error(f"Failed to get conversation {conversation_id}: {e}")
            return None
    
    def get_all_conversations(self) -> List:
        """Get all active conversations"""
        try:
            from .database import Conversation
            return self.db.query(Conversation).filter(
                Conversation.is_active == True
            ).order_by(Conversation.updated_at.desc()).all()
        except Exception as e:
            logger.error(f"Failed to get conversations: {e}")
            return []
    
    def add_message(self, conversation_id: int, query: str, answer: str, sources: List[str]):
        """Add a message to a conversation and update checkpoint"""
        try:
            from .database import Message
            # Create message
            message = Message(
                conversation_id=conversation_id,
                query=query,
                answer=answer,
                sources=json.dumps(sources) if sources else "[]"
            )
            self.db.add(message)
            
            # Update conversation timestamp
            conversation = self.get_conversation(conversation_id)
            if conversation:
                conversation.updated_at = datetime.now(timezone.utc)
            
            self.db.commit()
            self.db.refresh(message)
            
            # Update checkpoint
            self._update_conversation_checkpoint(conversation_id)
            
            logger.info(f"Added message to conversation {conversation_id}")
            return message
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to add message to conversation {conversation_id}: {e}")
            raise
    
    def save_checkpoint(self, conversation_id: int, checkpoint_data: Dict[str, Any]):
        """Save a conversation checkpoint"""
        try:
            from .database import Message, ChatCheckpoint
            # Get current message count
            message_count = self.db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).count()
            
            checkpoint = ChatCheckpoint(
                conversation_id=conversation_id,
                checkpoint_data=json.dumps(checkpoint_data),
                message_count=message_count
            )
            self.db.add(checkpoint)
            self.db.commit()
            self.db.refresh(checkpoint)
            
            logger.info(f"Saved checkpoint for conversation {conversation_id}")
            return checkpoint
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to save checkpoint for conversation {conversation_id}: {e}")
            raise
    
    def get_latest_checkpoint(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        """Get the latest checkpoint for a conversation"""
        try:
            from .database import ChatCheckpoint
            checkpoint = self.db.query(ChatCheckpoint).filter(
                ChatCheckpoint.conversation_id == conversation_id
            ).order_by(ChatCheckpoint.created_at.desc()).first()
            
            if checkpoint:
                return json.loads(checkpoint.checkpoint_data)
            return None
        except Exception as e:
            logger.error(f"Failed to get checkpoint for conversation {conversation_id}: {e}")
            return None
    
    def restore_conversation_state(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        """Restore conversation state from checkpoint"""
        try:
            from .database import Message
            conversation = self.get_conversation(conversation_id)
            if not conversation:
                return None
            
            # Get all messages for the conversation
            messages = self.db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(Message.created_at.asc()).all()
            
            # Build conversation state
            state = {
                "conversation_id": conversation.id,
                "title": conversation.title,
                "created_at": conversation.created_at.isoformat(),
                "updated_at": conversation.updated_at.isoformat(),
                "message_count": len(messages),
                "messages": []
            }
            
            for message in messages:
                state["messages"].append({
                    "id": message.id,
                    "query": message.query,
                    "answer": message.answer,
                    "sources": json.loads(message.sources) if message.sources else [],
                    "created_at": message.created_at.isoformat()
                })
            
            return state
        except Exception as e:
            logger.error(f"Failed to restore conversation state {conversation_id}: {e}")
            return None
    
    def delete_conversation(self, conversation_id: int) -> bool:
        """Soft delete a conversation"""
        try:
            conversation = self.get_conversation(conversation_id)
            if conversation:
                conversation.is_active = False
                self.db.commit()
                logger.info(f"Deleted conversation {conversation_id}")
                return True
            return False
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to delete conversation {conversation_id}: {e}")
            return False
    
    def get_conversation_history(self, conversation_id: int) -> List[Dict[str, Any]]:
        """Get formatted conversation history"""
        try:
            from .database import Message
            messages = self.db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(Message.created_at.asc()).all()
            
            history = []
            for message in messages:
                history.append({
                    "id": message.id,
                    "query": message.query,
                    "answer": message.answer,
                    "sources": json.loads(message.sources) if message.sources else [],
                    "created_at": message.created_at.isoformat()
                })
            
            return history
        except Exception as e:
            logger.error(f"Failed to get conversation history {conversation_id}: {e}")
            return []
    
    def _update_conversation_checkpoint(self, conversation_id: int):
        """Update checkpoint after adding a message"""
        try:
            state = self.restore_conversation_state(conversation_id)
            if state:
                self.save_checkpoint(conversation_id, state)
        except Exception as e:
            logger.error(f"Failed to update checkpoint for conversation {conversation_id}: {e}")
    
    def generate_conversation_title(self, first_query: str) -> str:
        """Generate a conversation title from the first query"""
        # Simple title generation - truncate and clean the query
        title = first_query.strip()
        if len(title) > 50:
            title = title[:47] + "..."
        
        # Remove special characters that might cause issues
        import re
        title = re.sub(r'[^\w\s-]', '', title)
        
        return title or "New Conversation"
