from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

class StreamQueryRequest(BaseModel):
    query: str
    stream: Optional[bool] = True

class StreamChunk(BaseModel):
    chunk: str
    sources: Optional[List[str]] = None
    done: bool = False