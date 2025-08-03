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
        gemini_api_key=settings.GOOGLE_GENERATIVE_AI_API_KEY
    )

@router.post("/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...), rag_service: RAGService = Depends(get_rag_service)):
    """Upload documents to the knowledge base."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate file types and sizes
    max_file_size = 10 * 1024 * 1024  # 10MB
    allowed_extensions = {'.pdf', '.txt', '.md', '.doc'}
    
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="File must have a filename")
        
        # Check file extension
        file_ext = '.' + file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file_ext} not supported. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Check file size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        if file_size > max_file_size:
            raise HTTPException(
                status_code=400, 
                detail=f"File {file.filename} is too large. Maximum size: 10MB"
            )
        
        if file_size == 0:
            raise HTTPException(
                status_code=400, 
                detail=f"File {file.filename} is empty"
            )
    
    try:
        documents_processed = await rag_service.process_documents(files)
        return UploadResponse(
            message=f"Successfully processed {len(files)} files and created {documents_processed} document chunks", 
            documents_processed=documents_processed
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")