import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "rag-assistant")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()