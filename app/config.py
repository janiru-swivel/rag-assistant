import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "rag-assistant")
    GOOGLE_GENERATIVE_AI_API_KEY: str = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY", "")
    
    # Performance optimization settings
    QUERY_TIMEOUT: int = int(os.getenv("QUERY_TIMEOUT", "25"))
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "2000"))
    SIMILARITY_SEARCH_K: int = int(os.getenv("SIMILARITY_SEARCH_K", "2"))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()