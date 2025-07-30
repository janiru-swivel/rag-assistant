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
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api_key)
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
            embedding=self.embeddings
        )

    async def process_documents(self, files: List[UploadFile]) -> int:
        documents = []
        for file in files:
            if file.filename.endswith(".pdf"):
                pdf_reader = PyPDF2.PdfReader(file.file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
                docs = self.text_splitter.create_documents([text], metadatas=[{"source": file.filename}])
                documents.extend(docs)
            else:
                content = await file.read()
                docs = self.text_splitter.create_documents([content.decode("utf-8")], metadatas=[{"source": file.filename}])
                documents.extend(docs)

        self.vector_store.add_documents(documents)
        return len(documents)

    async def query(self, query: str) -> Tuple[str, List[str]]:
        docs = self.vector_store.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = [doc.metadata.get("source", "Unknown") for doc in docs]
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer concisely and accurately based on the context provided."
        response = self.llm.invoke(prompt)
        return response.content, sources