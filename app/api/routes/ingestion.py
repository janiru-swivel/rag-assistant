from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from typing import List
from pydantic import BaseModel
from app.rag_service import RAGService
from app.config import settings

router = APIRouter(prefix="/ingestion", tags=["ingestion"])

class UploadResponse(BaseModel):
    message: str
    documents_processed: int

def get_rag_service():
    """Dependency injection for RAGService."""
    return RAGService(
        pinecone_api_key=settings.PINECONE_API_KEY,
        pinecone_index_name=settings.PINECONE_INDEX_NAME,
        gemini_api_key=settings.GOOGLE_API_KEY
    )

@router.post("/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...), rag_service: RAGService = Depends(get_rag_service)):
    """Upload documents to the knowledge base."""
    try:
        documents_processed = await rag_service.process_documents(files)
        return UploadResponse(message="Documents uploaded successfully", documents_processed=documents_processed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")