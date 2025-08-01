from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import auth, chat, history, ingestion, reference

app = FastAPI(
    title="RAG Knowledge Assistant",
    description="A Retrieval-Augmented Generation (RAG) based knowledge assistant using FastAPI, Pinecone, LangChain, and Gemini API.",
    version="1.0.0",
    openapi_tags=[
        {"name": "auth", "description": "Authentication endpoints"},
        {"name": "chat", "description": "Query the knowledge base"},
        {"name": "history", "description": "Manage chat history"},
        {"name": "ingestion", "description": "Upload documents to the knowledge base"},
        {"name": "reference", "description": "Retrieve document references"},
    ]
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include route modules
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(history.router)
app.include_router(ingestion.router)
app.include_router(reference.router)

@app.get("/health", tags=["health"])
async def health_check():
    """Check the health of the API server."""
    return {"status": "healthy"}