import os
import PyPDF2
import hashlib
import time
from typing import List, Tuple, Dict
from fastapi import UploadFile
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

class RAGService:
    def __init__(self, pinecone_api_key: str, pinecone_index_name: str, gemini_api_key: str):
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_index_name = pinecone_index_name
        self.gemini_api_key = gemini_api_key
        
        # Simple response cache for faster repeated queries
        self.query_cache: Dict[str, Tuple[str, List[str], float]] = {}
        self.cache_expiry = 300  # 5 minutes cache expiry
        
        # Set environment variable for Google API
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
        
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Optimize LLM for faster responses
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1,  # Lower temperature for faster, more focused responses
            max_tokens=300,  # Limit output tokens for faster responses
            timeout=15,  # 15-second timeout
            max_retries=1  # Fewer retries for faster failure handling
        )
        self._initialize_pinecone()

    def _initialize_pinecone(self):
        pc = Pinecone(api_key=self.pinecone_api_key)
        if self.pinecone_index_name not in pc.list_indexes().names():
            pc.create_index(
                name=self.pinecone_index_name,
                dimension=768,  # Gemini embedding dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        self.vector_store = PineconeVectorStore(
            index_name=self.pinecone_index_name,
            embedding=self.embeddings,
            pinecone_api_key=self.pinecone_api_key
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