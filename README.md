# RAG Assistant

A Retrieval-Augmented Generation (RAG) based knowledge assistant using FastAPI, Pinecone, LangChain, and Google Generative AI.

## Overview

This application provides a RAG assistant that can ingest various document formats, index them in a vector database (Pinecone), and allow users to query the knowledge base. The assistant uses Google's Generative AI (Gemini) to generate responses based on retrieved document chunks.

## API Endpoints

### Health Check

- `GET /health`: Check if the API server is healthy

### Document Ingestion

- `POST /ingestion/upload`: Upload documents to the knowledge base
  - Request: Form data with file(s) in the "files" field
  - Supported formats: PDF, TXT, MD, DOC
  - Max file size: 10MB

### Chat/Query

- `POST /chat/`: Query the knowledge base
  - Request: JSON with "query" field
  - Response: JSON with "answer" and "sources" fields

### Authentication

- Auth endpoints for user management

### History

- Endpoints for managing chat history

### Reference

- Endpoints for retrieving document references

## Testing

Several test scripts are available to verify functionality:

1. `test_comprehensive.py`: Tests all major endpoints
2. `test_chat.py`: Tests only the chat functionality
3. `test_any_file_ingestion.py`: Tests document ingestion with any file
4. `test_ingestion.py` and `test_ingestion_detailed.py`: Additional ingestion tests

To run the comprehensive tests:

```
source venv/bin/activate
python test_comprehensive.py
```

## Postman Collection

A Postman collection is available in the root directory for testing the API endpoints:

- `RAG_Assistant_API.postman_collection.json`
