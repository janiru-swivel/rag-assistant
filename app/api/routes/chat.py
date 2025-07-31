from fastapi import APIRouter, Depends, HTTPException
import asyncio
from app.models import QueryRequest, QueryResponse
from app.rag_service import RAGService
from app.config import settings

router = APIRouter(prefix="/chat", tags=["chat"])

def get_rag_service():
    """Dependency injection for RAGService."""
    return RAGService(
        pinecone_api_key=settings.PINECONE_API_KEY,
        pinecone_index_name=settings.PINECONE_INDEX_NAME,
        gemini_api_key=settings.GOOGLE_GENERATIVE_AI_API_KEY
    )

@router.post("/", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest, rag_service: RAGService = Depends(get_rag_service)):
    """Query the knowledge base with a user question."""
    try:
        print(f"Received query: {request.query}")
        
        # Add timeout wrapper for the entire query process
        answer, sources = await asyncio.wait_for(
            rag_service.query(request.query),
            timeout=25.0  # 25-second total timeout
        )
        
        print(f"Generated answer: {answer[:100]}...")
        return QueryResponse(answer=answer, sources=sources)
    except asyncio.TimeoutError:
        print("Query timed out")
        raise HTTPException(status_code=408, detail="Request timed out. Please try a shorter question or try again.")
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")