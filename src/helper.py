from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  
from typing import List
from langchain_core.documents import Document 
import os


# Extract Data From the PDF File
def load_pdf_file(data):
    """Load all PDF files from the given directory."""
    if not os.path.exists(data):
        raise FileNotFoundError(f" Folder not found: {data}")
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Simplify documents to only source metadata."""
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs


# Split the Data into Text Chunks
def text_split(extracted_data):
    """Split documents into smaller text chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# Download the Embeddings from HuggingFace
def download_hugging_face_embeddings():
    """Load HuggingFace sentence-transformer embeddings."""
    print(" Loading HuggingFace Embeddings (MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings
