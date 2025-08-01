import os
import PyPDF2
import hashlib
import time
import json
from typing import List, Tuple, Dict, AsyncGenerator
from fastapi import UploadFile
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.callbacks.base import BaseCallbackHandler
import pinecone

class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler that streams tokens as they are generated."""
    
    def __init__(self):
        self.tokens = []
        self.current_chunk = ""
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when a new token is generated."""
        self.tokens.append(token)
        self.current_chunk += token
    
    def get_current_text(self) -> str:
        """Get the current accumulated text."""
        return "".join(self.tokens)
    
    def clear(self):
        """Clear the accumulated tokens."""
        self.tokens = []
        self.current_chunk = ""

class RAGService:
    def __init__(self, pinecone_api_key: str, pinecone_index_name: str, gemini_api_key: str, mock_mode: bool = False):
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_index_name = pinecone_index_name
        self.gemini_api_key = gemini_api_key
        self.mock_mode = mock_mode
        
        # Simple response cache for faster repeated queries
        self.query_cache: Dict[str, Tuple[str, List[str], float]] = {}
        self.cache_expiry = 300  # 5 minutes cache expiry
        
        # Set environment variable for Google API
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
        
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        if not mock_mode:
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            # Optimize LLM for faster responses
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.1,  # Lower temperature for faster, more focused responses
                max_tokens=300,  # Limit output tokens for faster responses
                timeout=15,  # 15-second timeout
                max_retries=1  # Fewer retries for faster failure handling
            )
            try:
                self._initialize_pinecone()
            except Exception as e:
                print(f"Warning: Could not initialize Pinecone: {e}")
                print("Running in mock mode...")
                self.mock_mode = True
        else:
            print("Running in mock mode - no external services required")

    def _initialize_pinecone(self):
        pinecone.init(api_key=self.pinecone_api_key, environment="us-east-1-aws")
        if self.pinecone_index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.pinecone_index_name,
                dimension=768,  # Gemini embedding dimension
                metric="cosine"
            )
        
        index = pinecone.Index(self.pinecone_index_name)
        self.vector_store = PineconeVectorStore(
            index=index,
            embedding=self.embeddings,
            text_key="text"
        )

    def _process_pdf_file(self, content: bytes, filename: str) -> List[Document]:
        """Process a PDF file and return documents."""
        from io import BytesIO
        try:
            pdf_file = BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            
            if text.strip():
                return self.text_splitter.create_documents([text], metadatas=[{"source": filename}])
        except Exception as e:
            print(f"Error processing PDF file {filename}: {str(e)}")
        return []

    def _process_text_file(self, content: bytes, filename: str) -> List[Document]:
        """Process a text file and return documents."""
        try:
            text_content = content.decode("utf-8")
            if text_content.strip():
                return self.text_splitter.create_documents([text_content], metadatas=[{"source": filename}])
        except Exception as e:
            print(f"Error processing text file {filename}: {str(e)}")
        return []

    async def process_documents(self, files: List[UploadFile]) -> int:
        documents = []
        for file in files:
            if not file.filename:
                continue
                
            try:
                await file.seek(0)
                content = await file.read()
                
                if file.filename.endswith(".pdf"):
                    docs = self._process_pdf_file(content, file.filename)
                else:
                    docs = self._process_text_file(content, file.filename)
                
                documents.extend(docs)
            except Exception as e:
                print(f"Error processing file {file.filename}: {str(e)}")
                continue

        if documents:
            self.vector_store.add_documents(documents)
        return len(documents)

    async def query(self, query: str) -> Tuple[str, List[str]]:
        if self.mock_mode:
            return self._mock_query(query)
            
        try:
            # Add input validation
            if not query or not query.strip():
                return "Please provide a valid question.", []
            
            # Check cache first for faster responses
            query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
            current_time = time.time()
            
            if query_hash in self.query_cache:
                cached_answer, cached_sources, cache_time = self.query_cache[query_hash]
                if current_time - cache_time < self.cache_expiry:
                    print(f"Cache hit for query: {query[:50]}...")
                    return cached_answer, cached_sources
                else:
                    # Remove expired cache entry
                    del self.query_cache[query_hash]
            
            # Limit query length to prevent performance issues
            if len(query) > 500:  # Reduced from 1000 to 500
                query = query[:500]
            
            # Perform similarity search with reduced results for faster processing
            docs = self.vector_store.similarity_search(query, k=2)  # Reduced from 3 to 2
            
            if not docs:
                return "I don't have enough information to answer your question based on the uploaded documents.", []
            
            # Build context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in docs])
            sources = [doc.metadata.get("source", "Unknown") for doc in docs]
            
            # Limit context length more aggressively for faster processing
            if len(context) > 2000:  # Reduced from 4000 to 2000
                context = context[:2000] + "..."
            
            # Create a more concise prompt for faster responses
            prompt = f"""Answer briefly based on this context:

{context}

Question: {query}

Provide a concise answer (max 2-3 sentences)."""

            # Use async invoke with timeout
            response = await self.llm.ainvoke(prompt)
            
            answer = str(response.content)
            
            # Cache the response for future use
            self.query_cache[query_hash] = (answer, sources, current_time)
            
            # Keep cache size manageable (max 100 entries)
            if len(self.query_cache) > 100:
                oldest_key = min(self.query_cache.keys(), 
                               key=lambda k: self.query_cache[k][2])
                del self.query_cache[oldest_key]
            
            return answer, sources
            
        except Exception as e:
            print(f"Error in query processing: {str(e)}")
            return f"I encountered an error while processing your question: {str(e)}", []

    def _mock_query(self, query: str) -> Tuple[str, List[str]]:
        """Mock implementation for testing without external services."""
        mock_responses = {
            "ai": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence.",
            "machine learning": "Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.",
            "default": f"This is a mock response to your query: '{query}'. In a real implementation, this would be generated by the AI model based on your uploaded documents."
        }
        
        query_lower = query.lower()
        for key, response in mock_responses.items():
            if key in query_lower:
                return response, ["mock_document.pdf", "sample_file.txt"]
        
        return mock_responses["default"], ["mock_document.pdf"]

    async def stream_query(self, query: str) -> AsyncGenerator[Dict, None]:
        """Stream the response to a query chunk by chunk."""
        if self.mock_mode:
            async for chunk in self._mock_stream_query(query):
                yield chunk
            return
            
        try:
            # Add input validation
            if not query or not query.strip():
                yield {"chunk": "Please provide a valid question.", "sources": [], "done": True}
                return
            
            # Check cache first for faster responses
            query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
            current_time = time.time()
            
            if query_hash in self.query_cache:
                cached_answer, cached_sources, cache_time = self.query_cache[query_hash]
                if current_time - cache_time < self.cache_expiry:
                    print(f"Cache hit for query: {query[:50]}...")
                    # Stream cached response word by word for consistency
                    words = cached_answer.split()
                    for i, word in enumerate(words):
                        chunk = word + (" " if i < len(words) - 1 else "")
                        yield {"chunk": chunk, "sources": cached_sources if i == len(words) - 1 else None, "done": i == len(words) - 1}
                    return
                else:
                    # Remove expired cache entry
                    del self.query_cache[query_hash]
            
            # Limit query length to prevent performance issues
            if len(query) > 500:
                query = query[:500]
            
            # Perform similarity search with reduced results for faster processing
            docs = self.vector_store.similarity_search(query, k=2)
            
            if not docs:
                yield {"chunk": "I don't have enough information to answer your question based on the uploaded documents.", "sources": [], "done": True}
                return
            
            # Build context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in docs])
            sources = [doc.metadata.get("source", "Unknown") for doc in docs]
            
            # Limit context length more aggressively for faster processing
            if len(context) > 2000:
                context = context[:2000] + "..."
            
            # Create a more concise prompt for faster responses
            prompt = f"""Answer briefly based on this context:

{context}

Question: {query}

Provide a concise answer (max 2-3 sentences)."""

            # Create streaming LLM for this request
            streaming_llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.1,
                max_tokens=300,
                timeout=15,
                max_retries=1,
                streaming=True
            )

            # Stream the response
            accumulated_answer = ""
            async for chunk in streaming_llm.astream(prompt):
                content = str(chunk.content) if chunk.content else ""
                if content:
                    accumulated_answer += content
                    yield {"chunk": content, "sources": None, "done": False}
            
            # Final chunk with sources
            yield {"chunk": "", "sources": sources, "done": True}
            
            # Cache the complete response
            self.query_cache[query_hash] = (accumulated_answer, sources, current_time)
            
            # Keep cache size manageable (max 100 entries)
            if len(self.query_cache) > 100:
                oldest_key = min(self.query_cache.keys(), 
                               key=lambda k: self.query_cache[k][2])
                del self.query_cache[oldest_key]
            
        except Exception as e:
            print(f"Error in stream query processing: {str(e)}")
            yield {"chunk": f"I encountered an error while processing your question: {str(e)}", "sources": [], "done": True}

    async def _mock_stream_query(self, query: str) -> AsyncGenerator[Dict, None]:
        """Mock streaming implementation for testing."""
        import asyncio
        
        answer, sources = self._mock_query(query)
        words = answer.split()
        
        for i, word in enumerate(words):
            await asyncio.sleep(0.1)  # Simulate streaming delay
            chunk = word + (" " if i < len(words) - 1 else "")
            is_done = i == len(words) - 1
            yield {
                "chunk": chunk, 
                "sources": sources if is_done else None, 
                "done": is_done
            }