# Import necessary libraries
from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings


import os
import waybackpy
from datetime import datetime
import requests
from io import BytesIO

# Set your OpenAI API key (replace with your key or use Sentence-BERT)
GOOGLE_API_KEY='AIzaSyAkRTto6ZaF2TzCByE6RK1UgRMuWznBk3Y'
# Set the Google API key from environment variable
google_api_key = os.getenv(GOOGLE_API_KEY)
if not google_api_key:
    raise ValueError("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")

# Step 1.1: Collect Documents (Local PDFs + Web Archives)
# Load a local PDF file
file_path = "sample_paper.pdf"  # Replace with your file path
loader = OnlinePDFLoader(file_path)
local_documents = loader.load()

# Function to fetch archived web content from the Wayback Machine
def fetch_wayback_content(url, year=None):
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    wayback = waybackpy.Url(url, user_agent)
    
    # Use a specific year if provided, otherwise get the most recent archive
    if year:
        archive = wayback.near(year=year)
    else:
        archive = wayback.get()
    
    # Fetch the archived content
    response = requests.get(archive.archive_url)
    if "application/pdf" in response.headers.get("Content-Type", ""):
        # Handle PDF content
        pdf_content = BytesIO(response.content)
        loader = OnlinePDFLoader(pdf_content)
        return loader.load()
    else:
        # Handle HTML/text content (simplified: extract raw text)
        return [{"page_content": response.text, "metadata": {"source": archive.archive_url}}]

# Collect web archive documents (example URL)
web_url = "https://arxiv.org/pdf/1706.03762.pdf"  # Example research paper URL
web_documents = fetch_wayback_content(web_url, year=2020)  # Fetch from 2020 archive
all_documents = local_documents + web_documents

# Step 1.2: Preprocess - Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Adjust based on your needs
    chunk_overlap=50  # Overlap to maintain context
)

# Split all documents (local + web) into smaller chunks
chunks = text_splitter.split_documents(all_documents)
print(f"Split into {len(chunks)} chunks.")

# Step 1.3: Choose a Vector Database - Set up Qdrant
qdrant_client = QdrantClient(
    url="https://db5344d0-726e-4714-bee8-4bd95b83802b.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Kcqp7qExFyhDzXknNDdX8QqC1JL6psQLXviJ1PjI1bk",
)
# Create a collection in Qdrant
collection_name = "research_assistant"
qdrant_client.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
)

# Step 1.4: Store Embeddings
# Function to generate embeddings using OpenAI
def get_openai_embedding(text):
    response = GoogleGenerativeAIEmbeddings()
    return response["data"][0]["embedding"]

# Alternative: Use Sentence-BERT (uncomment if preferred)
# from sentence_transformers import SentenceTransformer
# embedder = SentenceTransformer('all-MiniLM-L6-v2')
# def get_sentencebert_embedding(text):
#     return embedder.encode(text).tolist()

# Prepare and upload chunks to Qdrant
points = []
for i, chunk in enumerate(chunks):
    chunk_text = chunk.page_content
    source = chunk.metadata.get("source", "unknown")  # Get source from metadata
    
    # Generate embedding
    embedding =GoogleGenerativeAIEmbeddings(model="models/embedding-001",)
    embedding.embed_query(chunk_text)
    
    # Create a point for Qdrant
    points.append(
        models.PointStruct(
            id=i,
            vector=embedding,
            payload={"text": chunk_text, "source": source}
        )
    )

# Upload all points to Qdrant
qdrant_client.upsert(
    collection_name=collection_name,
    points=points
)

print(f"Stored {len(points)} chunks in Qdrant collection '{collection_name}'.")