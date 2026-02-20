from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
import json
import glob
import logging
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from typing import List, Dict, Any
import re
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
class ResearchExtractor:
    """Class to extract research topics and goals from the knowledge graph."""

    SEMANTIC_TEMPLATE = (
        "Analyze the following research paper text and extract key topics "
        "and research goals for a knowledge graph:\n\n"
        "Research Text:\n{text}\n\n"
        "Provide only two lists in your response:\n\n"
        "1. A list of main research topics (subjects, concepts, specialized terminology)\n"
        "2. A list of objective research goals (starting with action verbs)\n\n"
        "Format:\n"
        "Topics:\n- [Topic 1]\n- [Topic 2]\n...\n\n"
        "Goals:\n- [Goal 1]\n- [Goal 2]\n..."
    )

    def __init__(self, qdrant_client: QdrantClient, collection_name: str, llm=None):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        self.topic_keywords = ["research", "study", "investigate", "examine", "analyze", "explore"]
        self.goal_keywords = ["aim", "objective", "purpose", "goal", "intention", "target"]

        # Initialize LLM chain for semantic analysis when llm is provided
        if self.llm is not None:
            self.semantic_chain = LLMChain(
                llm=self.llm,
                prompt=PromptTemplate(
                    input_variables=["text"],
                    template=self.SEMANTIC_TEMPLATE,
                ),
            )
        else:
            self.semantic_chain = None



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

def load_chunks_from_jsonl(chunks_dir: str = "chunks") -> List[Dict[str, Any]]:
    """Load chunks from JSONL files.

    :param chunks_dir: Directory containing chunk JSONL files
    :type chunks_dir: str
    :return: List of chunk dictionaries
    :rtype: List[Dict[str, Any]]
    """
    chunks = []
    chunk_files = glob.glob(os.path.join(chunks_dir, "*-chunks.jsonl"))

    for file_path in chunk_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line)
                chunks.append(chunk)
    return chunks


def build_word2vec_model(chunks: List[Dict[str, Any]], vector_size: int = 100) -> Word2Vec:
    """Train a Word2Vec model on chunk texts.

    :param chunks: List of chunk dictionaries with a 'chunk' key
    :param vector_size: Dimensionality of the word vectors
    :return: Trained Word2Vec model
    """
    sentences = [simple_preprocess(chunk['chunk']) for chunk in chunks]
    return Word2Vec(sentences, vector_size=vector_size, window=5, min_count=1, workers=4)


def create_qdrant_points(chunks: List[Dict[str, Any]], word2vec_model: Word2Vec) -> List[models.PointStruct]:
    """Create Qdrant point structs from chunks using Word2Vec embeddings.

    :param chunks: List of chunk dictionaries
    :param word2vec_model: Trained Word2Vec model
    :return: List of PointStruct objects ready for upload
    """
    points = []
    for i, chunk in enumerate(chunks):
        tokens = simple_preprocess(chunk['chunk'])
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
    return points


def upload_points_to_qdrant(
    qdrant_client: QdrantClient,
    collection_name: str,
    points: List[models.PointStruct],
    batch_size: int = 100,
) -> int:
    """Upload point structs to a Qdrant collection in batches.

    :param qdrant_client: Initialized QdrantClient instance
    :param collection_name: Name of the target collection
    :param points: Points to upload
    :param batch_size: Number of points per upload batch
    :return: Number of points uploaded
    """
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        qdrant_client.upsert(collection_name=collection_name, points=batch)
    return len(points)


def initialize_knowledge_graph(
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    collection_name: str = "research_assistant",
    vector_size: int = 100,
) -> tuple:
    """Initialize the Qdrant-backed knowledge graph.

    Reads credentials from environment variables when not supplied explicitly:
      - ``QDRANT_URL``
      - ``QDRANT_API_KEY``

    :return: Tuple of (QdrantClient, collection_name, ResearchExtractor)
    """
    url = qdrant_url or os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = qdrant_api_key or os.environ.get("QDRANT_API_KEY")

    client = QdrantClient(url=url, api_key=api_key)
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )
    extractor = ResearchExtractor(client, collection_name)
    return client, collection_name, extractor