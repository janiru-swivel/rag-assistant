from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter(prefix="/reference", tags=["reference"])

class Reference(BaseModel):
    source: str
    content: str

@router.get("/", response_model=List[Reference])
async def get_references():
    """Retrieve references from the knowledge base."""
    # Placeholder: In a real implementation, fetch from Pinecone or a database
    return [{"source": "sample.pdf", "content": "Sample content from document"}]