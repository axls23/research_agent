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
from typing import List, Dict, Any
import re
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class ResearchExtractor:
    """Class to extract research topics and goals from the knowledge graph."""
    
    def __init__(self, qdrant_client: QdrantClient, collection_name: str, llm):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.llm=llm
        self.topic_keywords = ["research", "study", "investigate", "examine", "analyze", "explore"]
        self.goal_keywords = ["aim", "objective", "purpose", "goal", "intention", "target"]
         
        # Initialize LLM chain for semantic analysis
        self.semantic_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=knowldege_base.get_chunks,
                template= template            )
        )
        template=('''from langchain.prompts import PromptTemplate 
        Analyze the following research paper text and extract key topics and research goals for a knowledge graph:

        Research Text:
        {self.dataset}

        Provide only two lists in your response:

        1. A list of main research topics (subjects, concepts, specialized terminology)
        2. A list of objective research goals (starting with action verbs)

        Format:
        Topics:
        - [Topic 1]
        - [Topic 2]
        ...

        Goals:
        - [Goal 1]
        - [Goal 2]
        ...''')



    async def analyze_chunk_semantics(self, chunk: str) -> Dict[str, List[str]]:
        """Use LLM to analyze chunk semantics and extract topics/goals."""
        try:
            result = await self.semantic_chain.arun(chunk)
            return json.loads(result)
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {str(e)}")
            return {"topics": [], "goals": []}

    async def build_semantic_graph(self, chunks: List[str]) -> Dict[str, Any]:
        """Build semantic graph from chunks using LLM analysis."""
        nodes = set()
        edges = []
        
        for chunk in chunks:
            analysis = await self.analyze_chunk_semantics(chunk)
            topics = analysis.get("topics", [])
            goals = analysis.get("goals", [])
            
            # Add nodes
            nodes.update(topics)
            nodes.update(goals)
            
            # Add edges between topics and goals
            for topic in topics:
                for goal in goals:
                    edges.append({"from": topic, "to": goal, "type": "topic_goal"})
        
        return {
            "nodes": list(nodes),
            "edges": edges
        }

    def extract_topics_and_goals(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Extract research topics and goals from the knowledge graph."""
        # Search for relevant documents
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=self._get_query_embedding(query),
            limit=limit
        )
        
        topics = set()
        goals = set()
        
        for result in search_results:
            text = result.payload["text"]
            metadata = result.payload["metadata"]
            
            # Extract topics from title and text
            if metadata.get("title"):
                topics.update(self._extract_topics(metadata["title"]))
            topics.update(self._extract_topics(text))
            
            # Extract goals from text
            goals.update(self._extract_goals(text))
        
        return {
            "topics": list(topics),
            "goals": list(goals)
        }
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract research topics from text."""
        topics = []
        # Look for topic indicators in sentences
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in self.topic_keywords):
                # Extract the main subject of the sentence
                topic = self._clean_text(sentence)
                if topic:
                    topics.append(topic)
        return topics
    
    def _extract_goals(self, text: str) -> List[str]:
        """Extract research goals from text."""
        goals = []
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in self.goal_keywords):
                goal = self._clean_text(sentence)
                if goal:
                    goals.append(goal)
        return goals
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove common prefixes and clean up
        text = text.strip()
        text = re.sub(r'^(The|This|Our|We|In this|Here|To|For)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query text."""
        # Use Word2Vec to get query embedding
        tokens = simple_preprocess(query)
        vectors = [self.word2vec_model.wv[word] for word in tokens if word in self.word2vec_model.wv]
        if vectors:
            return np.mean(vectors, axis=0).tolist()
        return [0.0] * 100  # Return zero vector if no tokens found

# Initialize Qdrant client and collection
apikey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.GF3HNT8RlmH4b5x2xAQxIvzzBXRXIJDVrYJqCo0nIDk'
qdrant_client = QdrantClient(
    url="https://db19c5fd-26bb-4c9a-8693-c99936abc5f1.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key=apikey
)
if not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = getpass.getpass("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 ")

    

collection_name = "research_assistant"
qdrant_client.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=100, distance=models.Distance.COSINE)
)

# Initialize research extractor
research_extractor = ResearchExtractor(qdrant_client, collection_name)

# Example usage:
# results = research_extractor.extract_topics_and_goals("machine learning research")
# print("Topics:", results["topics"])
# print("Goals:", results["goals"])

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