import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Load environment variables from .env
load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    print("Error: QDRANT_URL or QDRANT_API_KEY not found in environment.")
    print("Make sure your .env file is populated.")
    exit(1)

print(f"Connecting to Qdrant Cloud at {QDRANT_URL}...")
try:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Check existing collections
    collections = client.get_collections()
    print(f"Successfully connected! Current collections: {[c.name for c in collections.collections]}")
    
    collection_name = "research_entities"
    
    if client.collection_exists(collection_name):
        print(f"Collection '{collection_name}' already exists. Recreating it to ensure clean state...")
        client.delete_collection(collection_name=collection_name)
    else:
        print(f"Creating collection '{collection_name}'...")

    # Initialize collection for BAAI/bge-small-en-v1.5
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=384,  # BGE-small dimension size
            distance=models.Distance.COSINE,
        ),
    )
    
    print("\n[SUCCESS] Setup complete! Qdrant Cloud is ready to receive vectors.")

except Exception as e:
    print(f"\n[ERROR] Error connecting to or configuring Qdrant: {e}")
