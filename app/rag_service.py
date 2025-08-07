import os
import PyPDF2
from typing import List, Tuple
from fastapi import UploadFile
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

class RAGService:
    def __init__(self, pinecone_api_key: str, pinecone_index_name: str, gemini_api_key: str):
        # Validate API keys
        if not pinecone_api_key or pinecone_api_key.strip() == "":
            raise ValueError("Pinecone API key is missing or empty")
        if not gemini_api_key or gemini_api_key.strip() == "" or "google_generative_ai_api_key" in gemini_api_key or "AIzaSyDummy" in gemini_api_key:
            raise ValueError("Google Generative AI API key is missing or not properly configured")
        
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_index_name = pinecone_index_name
        self.gemini_api_key = gemini_api_key
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        try:
            # Set as environment variable as the library prefers it this way
            os.environ["GOOGLE_API_KEY"] = gemini_api_key
            
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            # Optimize LLM for faster responses
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                temperature=0.1,  # Lower temperature for more consistent responses
                max_tokens=500,   # Limit response length for faster generation
                timeout=30        # Set timeout to prevent hanging
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize Google Generative AI services. Check the API key: {str(e)}")
        
        self._initialize_pinecone()

    def _initialize_pinecone(self):
        pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Check if index exists and has correct dimensions
        existing_indexes = pc.list_indexes().names()
        if self.pinecone_index_name in existing_indexes:
            # Get index info to check dimensions
            index_info = pc.describe_index(self.pinecone_index_name)
            index_dimension = index_info['dimension']
            required_dimension = 768  # Google Generative AI embedding-001 dimension
            
            if index_dimension != required_dimension:
                print(f"Warning: Existing index has dimension {index_dimension}, but embeddings need {required_dimension}")
                print(f"Deleting and recreating index '{self.pinecone_index_name}' with correct dimensions...")
                pc.delete_index(self.pinecone_index_name)
                # Wait a moment for deletion to complete
                import time
                time.sleep(5)
        
        # Create index if it doesn't exist (or was just deleted)
        if self.pinecone_index_name not in pc.list_indexes().names():
            print(f"Creating Pinecone index '{self.pinecone_index_name}' with dimension 768...")
            pc.create_index(
                name=self.pinecone_index_name,
                dimension=768,  # Correct dimension for Gemini embedding-001
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print("Index created successfully!")
        
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
        """Process uploaded documents and add them to the vector store."""
        if not files:
            raise ValueError("No files provided for processing")
        
        documents = []
        errors = []
        
        for file in files:
            try:
                docs = await self._process_single_file(file, errors)
                if docs:
                    documents.extend(docs)
            except Exception as e:
                error_msg = f"Error processing file {file.filename}: {str(e)}"
                print(error_msg)
                errors.append(error_msg)

        if not documents:
            error_summary = "; ".join(errors) if errors else "No documents could be processed"
            raise ValueError(f"Failed to process any documents. Errors: {error_summary}")

        try:
            print(f"Adding {len(documents)} document chunks to vector store...")
            self.vector_store.add_documents(documents)
            print("Successfully added documents to vector store")
        except Exception as e:
            raise ValueError(f"Failed to add documents to vector store: {str(e)}")
        
        return len(documents)
    
    async def _process_single_file(self, file: UploadFile, errors: List[str]) -> List[Document]:
        """Process a single uploaded file and return documents."""
        if not file.filename:
            errors.append("File with no filename provided")
            return []
            
        print(f"Processing file: {file.filename}")
        await file.seek(0)
        content = await file.read()
        
        if not content:
            errors.append(f"File {file.filename} is empty")
            return []
        
        # Support multiple file types
        filename_lower = file.filename.lower()
        if filename_lower.endswith(".pdf"):
            docs = self._process_pdf_file(content, file.filename)
        elif filename_lower.endswith((".txt", ".md", ".doc")):
            docs = self._process_text_file(content, file.filename)
        else:
            # Try to process as text anyway
            docs = self._process_text_file(content, file.filename)
        
        if docs:
            print(f"Successfully processed {file.filename}: {len(docs)} chunks created")
        else:
            errors.append(f"No content extracted from {file.filename}")
            
        return docs

    async def query(self, query: str) -> Tuple[str, List[str]]:
        try:
            # Add input validation
            if not query or not query.strip():
                return "Please provide a valid question.", []
            
            # Limit query length to prevent performance issues
            if len(query) > 1000:
                query = query[:1000]
            
            # Perform similarity search with timeout handling
            docs = self.vector_store.similarity_search(query, k=3)
            
            if not docs:
                return "I don't have enough information to answer your question based on the uploaded documents.", []
            
            # Build context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in docs])
            sources = [doc.metadata.get("source", "Unknown") for doc in docs]
            
            # Limit context length to prevent token limits
            if len(context) > 4000:
                context = context[:4000] + "..."
            
            # Create a more structured prompt
            prompt = f"""Based on the following context, please answer the question concisely and accurately.

Context:
{context}

Question: {query}

Please provide a clear and concise answer based only on the information provided in the context above. If the context doesn't contain enough information to answer the question, please say so."""

            # Use async invoke with timeout
            response = await self.llm.ainvoke(prompt)
            
            return str(response.content), sources
            
        except Exception as e:
            print(f"Error in query processing: {str(e)}")
            return f"I encountered an error while processing your question: {str(e)}", []