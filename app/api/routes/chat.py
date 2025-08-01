from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
import asyncio
import json
from app.models import QueryRequest, QueryResponse, StreamQueryRequest
from app.rag_service import RAGService
from app.config import settings

router = APIRouter(prefix="/chat", tags=["chat"])

def get_rag_service():
    """Dependency injection for RAGService."""
    return RAGService(
        pinecone_api_key=settings.PINECONE_API_KEY,
        pinecone_index_name=settings.PINECONE_INDEX_NAME,
        gemini_api_key=settings.GOOGLE_GENERATIVE_AI_API_KEY,
        mock_mode=True  # Enable mock mode for testing without external services
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

@router.post("/stream")
async def stream_query_knowledge_base(request: StreamQueryRequest, rag_service: RAGService = Depends(get_rag_service)):
    """Stream the response from the knowledge base with a user question."""
    try:
        print(f"Received streaming query: {request.query}")
        
        async def generate_stream():
            try:
                async for chunk_data in rag_service.stream_query(request.query):
                    # Format as Server-Sent Events
                    json_data = json.dumps(chunk_data)
                    yield f"data: {json_data}\n\n"
                
                # Send final event to indicate completion
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                error_data = {
                    "chunk": f"Error: {str(e)}", 
                    "sources": [], 
                    "done": True
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
        
    except Exception as e:
        print(f"Error in streaming chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing streaming query: {str(e)}")