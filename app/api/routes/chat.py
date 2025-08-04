from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.models import QueryRequest, QueryResponse
from app.rag_service import RAGService
from app.config import settings
from app.database import get_db # pyright: ignore[reportMissingImports]
from app.checkpointer import ChatCheckpointer # pyright: ignore[reportMissingImports]
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

def get_rag_service():
    """Dependency injection for RAGService."""
    return RAGService(
        pinecone_api_key=settings.PINECONE_API_KEY,
        pinecone_index_name=settings.PINECONE_INDEX_NAME,
        gemini_api_key=settings.GOOGLE_GENERATIVE_AI_API_KEY
    )

@router.post("/", response_model=QueryResponse)
async def query_knowledge_base(
    request: QueryRequest, 
    rag_service: RAGService = Depends(get_rag_service),
    db: Session = Depends(get_db)
):
    """Query the knowledge base with a user question."""
    try:
        checkpointer = ChatCheckpointer(db)
        
        # Get or create conversation
        conversation = None
        if request.conversation_id:
            conversation = checkpointer.get_conversation(request.conversation_id)
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")
        else:
            # Create new conversation with auto-generated title
            title = checkpointer.generate_conversation_title(request.query)
            conversation = checkpointer.create_conversation(title)
        
        # Query the RAG service
        answer, sources = await rag_service.query(request.query)
        
        # Save message to database with checkpoint
        message = checkpointer.add_message(
            conversation.id, 
            request.query, 
            answer, 
            sources
        )
        
        return QueryResponse(
            id=message.id,
            query=request.query,
            answer=answer,
            sources=sources,
            references=sources,  # For Postman compatibility
            created_at=message.created_at.isoformat(),
            conversation_id=conversation.id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@router.post("/conversation/{conversation_id}", response_model=QueryResponse)
async def query_in_conversation_context(
    conversation_id: int,
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
    db: Session = Depends(get_db)
):
    """Query the knowledge base within a specific conversation context."""
    try:
        checkpointer = ChatCheckpointer(db)
        
        # Get conversation
        conversation = checkpointer.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get conversation history for context
        history = checkpointer.get_conversation_history(conversation_id)
        
        # Build context from previous messages
        context_messages = []
        for msg in history[-5:]:  # Last 5 messages for context
            context_messages.append(f"Q: {msg['query']}\nA: {msg['answer']}")
        
        # Enhance query with context if available
        enhanced_query = request.query
        if context_messages:
            context = "\n\n".join(context_messages)
            enhanced_query = f"Context from previous conversation:\n{context}\n\nCurrent question: {request.query}"
        
        # Query the RAG service
        answer, sources = await rag_service.query(enhanced_query)
        
        # Save message to database with checkpoint
        message = checkpointer.add_message(
            conversation_id, 
            request.query,  # Save original query, not enhanced
            answer, 
            sources
        )
        
        return QueryResponse(
            id=message.id,
            query=request.query,
            answer=answer,
            sources=sources,
            references=sources,  # For Postman compatibility
            created_at=message.created_at.isoformat(),
            conversation_id=conversation_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query in conversation {conversation_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")