from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as PineconeVectorStore  


# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# Step 1: Load and process PDF files
extracted_data = load_pdf_file(data="data/")
filtered_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filtered_data)

# Step 2: Load embeddings model
embeddings = download_hugging_face_embeddings()

# Step 3: Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Step 4: Create index if it doesn’t exist
index_name = "medical-chatbot"

# New method for Pinecone v3
existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if index_name not in existing_indexes:
    print(f"🔧 Creating new index '{index_name}' ...")
    pc.create_index(
        name=index_name,
        dimension=384,  # all-MiniLM-L6-v2 output size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
else:
    print(f" Using existing index '{index_name}'.")

# Step 5: Connect to index
index = pc.Index(index_name)

# Step 6: Upload (upsert) documents
print(" Uploading embeddings to Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print(" Successfully stored all documents in Pinecone!")
