from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_community.tools import arxiv
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.chat_models import init_chat_model
import os
import json
import glob
from tavily import TavilyClient
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Step 1: Set up Qdrant client
apikey='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.FXw9K_pUdMyKUCpkGzOHtyZtwi7-SA9_AQxFlXIGU4I'
qdrant_client = QdrantClient(
    url="https://db5344d0-726e-4714-bee8-4bd95b83802b.europe-west3-0.gcp.cloud.qdrant.io",
    api_key=apikey
)

# Create or recreate collection
collection_name = "research_assistant"
qdrant_client.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=100, distance=models.Distance.COSINE)  # Using 100 dimensions for Word2Vec
)

# Step 2: Load chunks from JSONL files
def load_chunks_from_jsonl():
    chunks = []
    # Only process files ending with -chunks.jsonl
    chunk_files = glob.glob("chunks/*-chunks.jsonl")
    
    for file_path in chunk_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line)
                chunks.append(chunk)
    return chunks

# Load all chunks
chunks = load_chunks_from_jsonl()
print(f"Loaded {len(chunks)} chunks from JSONL files")

# Step 3: Prepare text for Word2Vec
sentences = []
for chunk in chunks:
    # Tokenize the text using simple_preprocess
    tokens = simple_preprocess(chunk['chunk'])  # Using the 'chunk' key
    sentences.append(tokens)

# Train Word2Vec model
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Step 4: Create embeddings and store in Qdrant
points = []
for i, chunk in enumerate(chunks):
    # Get tokens for the chunk
    tokens = simple_preprocess(chunk['chunk'])  # Using the 'chunk' key
    
    # Calculate average vector for all words in the chunk
    vectors = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
    if vectors:
        embedding = np.mean(vectors, axis=0).tolist()
        
        points.append(
            models.PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "text": chunk['chunk'],
                    "source": chunk.get('source', 'unknown'),
                    "metadata": {
                        "doi": chunk.get('doi', ''),
                        "chunk_id": chunk.get('chunk-id', ''),
                        "title": chunk.get('title', '')
                    }
                }
            )
        )

# Upload to Qdrant
batch_size = 100
for i in range(0, len(points), batch_size):
    batch = points[i:i + batch_size]
    qdrant_client.upsert(collection_name=collection_name, points=batch)
    print(f"Uploaded batch {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size}")

print(f"Successfully stored {len(points)} chunks in Qdrant collection '{collection_name}'.")