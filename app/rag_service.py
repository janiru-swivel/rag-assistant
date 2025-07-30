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
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_index_name = pinecone_index_name
        self.gemini_api_key = gemini_api_key
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
        # Optimize LLM for faster responses
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=gemini_api_key,
            temperature=0.1,  # Lower temperature for more consistent responses
            max_tokens=500,   # Limit response length for faster generation
            timeout=30        # Set timeout to prevent hanging
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