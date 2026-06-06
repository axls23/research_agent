"""
core/nodes/retrieval_node.py
============================
LangGraph node for semantic retrieval using the SIMD-accelerated turbovec engine.
Replaces previous heavy vector databases (e.g. ChromaDB) for in-memory, highly compressed search.
"""

import os
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from turbovec import IdMapIndex
from turbovec.langchain import TurbovecStore

from core.state import ResearchState, append_audit

# Local index location
INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "turbovec_index.tvim")

def get_embeddings() -> OllamaEmbeddings:
    """Configure local embedding function for the air-gapped harness."""
    return OllamaEmbeddings(model="nomic-embed-text")

def load_or_create_turbovec_store(docs: Optional[List[Document]] = None) -> TurbovecStore:
    """
    Standard boilerplate for index management with turbovec.
    Loads an existing .tvim index from disk if it exists, otherwise creates a new one.
    """
    embeddings = get_embeddings()
    
    if os.path.exists(INDEX_PATH) and docs is None:
        # turbovec index load via IdMapIndex
        index = IdMapIndex.load(INDEX_PATH)
        return TurbovecStore(index=index, embedding=embeddings)
    elif docs is not None:
        store = TurbovecStore.from_documents(docs, embeddings)
        # Save index locally so we don't have to re-embed on every run
        store.index.save(INDEX_PATH)
        return store
    else:
        raise ValueError("No existing index found and no documents provided to create one.")

def pre_filter_academic_papers(store: TurbovecStore, query: str, allowed_paper_ids: List[str], k: int = 5) -> List[Document]:
    """
    GraphRAG Pre-filtering Helper.
    Demonstrates using turbovec's SIMD-accelerated allowlist feature.
    Passes specific document IDs to restrict semantic search to a bounded subset.
    """
    # Utilizing turbovec's native SIMD-accelerated allowlist parameter
    results = store.similarity_search(
        query,
        k=k,
        allowlist=allowed_paper_ids
    )
    return results

def retrieve_research_context(state: ResearchState) -> dict:
    """
    LangGraph node function to retrieve relevant context using turbovec.
    Extracts the user's latest query, queries the turbovec index, and
    appends the retrieved document chunks back to the state's context window.
    """
    queries = state.get("search_queries", [])
    if not queries:
        return {"current_node": "retrieval_node"}
    
    latest_query = queries[-1]
    
    try:
        store = load_or_create_turbovec_store()
    except ValueError as e:
        # Gracefully handle missing index
        return {"current_node": "retrieval_node"}
    
    # Optional GraphRAG pre-filtering: Restrict to explicitly included papers
    included_papers = [p["paper_id"] for p in state.get("papers", []) if p.get("included", True)]
    
    # High-recall target (k=5) semantic search
    if included_papers:
        retrieved_docs = pre_filter_academic_papers(store, latest_query, allowed_paper_ids=included_papers, k=5)
    else:
        retrieved_docs = store.similarity_search(latest_query, k=5)
    
    # Append the retrieved document chunks back into the state's context window
    new_chunks = []
    for doc in retrieved_docs:
        new_chunks.append({
            "chunk_id": doc.metadata.get("chunk_id", "unknown"),
            "paper_id": doc.metadata.get("paper_id", "unknown"),
            "text": doc.page_content,
            "token_count": len(doc.page_content.split()),
            "page_range": doc.metadata.get("page_range")
        })
    
    existing_chunks = state.get("chunks", [])
    updated_chunks = existing_chunks + new_chunks
    
    audit_log = append_audit(
        state,
        agent="retrieval_node",
        action="turbovec_semantic_search",
        inputs={"query": latest_query, "k": 5, "allowlist_size": len(included_papers)},
        output_summary=f"Retrieved {len(retrieved_docs)} chunks from Turbovec.",
        provenance={"model": "nomic-embed-text", "retrieved_ids": [c["chunk_id"] for c in new_chunks]}
    )
    
    return {
        "current_node": "retrieval_node",
        "chunks": updated_chunks,
        "audit_log": audit_log
    }
