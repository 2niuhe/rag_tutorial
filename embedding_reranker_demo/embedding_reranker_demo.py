#!/usr/bin/env python3
"""
Demo script to test BGE-M3 embedding model and BGE-Reranker-v2 reranker model
using sentence-transformers library on a corpus of OpenStack Nova command documentation.
"""

import os
import glob
import time
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.cross_encoder import CrossEncoder

# Configuration
CORPUS_DIR = "../corpus"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2"
TOP_K_RETRIEVAL = 20  # Number of documents to retrieve with embeddings
TOP_K_RERANKED = 5    # Number of documents to show after reranking

class Document:
    def __init__(self, doc_id: str, content: str, filepath: str):
        self.id = doc_id
        self.content = content
        self.filepath = filepath
        
    def __str__(self):
        return f"Document(id={self.id}, filepath={self.filepath})"

def load_corpus(corpus_dir: str) -> List[Document]:
    """Load all text documents from the corpus directory."""
    documents = []
    files = glob.glob(os.path.join(corpus_dir, "*.txt"))
    
    print(f"Loading {len(files)} documents from {corpus_dir}...")
    for i, filepath in enumerate(tqdm(files)):
        filename = os.path.basename(filepath)
        doc_id = f"doc_{i}"
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        documents.append(Document(doc_id, content, filename))
    
    return documents

def create_embeddings(model: SentenceTransformer, documents: List[Document]) -> np.ndarray:
    """Create embeddings for all documents in the corpus."""
    print(f"Creating embeddings using {model.get_sentence_embedding_dimension()}-dimensional vectors...")
    
    contents = [doc.content for doc in documents]
    embeddings = model.encode(contents, show_progress_bar=True, convert_to_numpy=True)
    
    return embeddings

def semantic_search(query: str, 
                    documents: List[Document], 
                    embeddings: np.ndarray, 
                    embedding_model: SentenceTransformer,
                    reranker_model: CrossEncoder,
                    top_k_retrieval: int = TOP_K_RETRIEVAL,
                    top_k_reranked: int = TOP_K_RERANKED) -> List[Tuple[Document, float]]:
    """
    Perform semantic search with embeddings and reranking.
    
    Args:
        query: The search query
        documents: List of Document objects
        embeddings: Pre-computed document embeddings
        embedding_model: SentenceTransformer model for embedding generation
        reranker_model: CrossEncoder model for reranking
        top_k_retrieval: Number of documents to retrieve with embeddings
        top_k_reranked: Number of documents to return after reranking
        
    Returns:
        List of (document, score) tuples for the top results
    """
    # Step 1: Embed the query
    start_time = time.time()
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    
    # Step 2: Compute cosine similarities between query and all documents
    cos_scores = util.cos_sim(query_embedding, embeddings)[0]
    
    # Step 3: Get top-k documents based on cosine similarity
    top_k_indices = np.argsort(-cos_scores)[:top_k_retrieval]
    top_k_docs = [documents[idx] for idx in top_k_indices]
    top_k_scores = [cos_scores[idx].item() for idx in top_k_indices]
    
    embedding_time = time.time() - start_time
    print(f"Embedding search took {embedding_time:.2f} seconds")
    
    # Step 4: Rerank the top-k documents using the reranker model
    start_time = time.time()
    rerank_pairs = [(query, doc.content) for doc in top_k_docs]
    rerank_scores = reranker_model.predict(rerank_pairs)
    
    # Step 5: Sort by reranker scores and get the top results
    reranked_indices = np.argsort(-rerank_scores)[:top_k_reranked]
    reranked_docs = [(top_k_docs[idx], rerank_scores[idx]) for idx in reranked_indices]
    
    reranking_time = time.time() - start_time
    print(f"Reranking took {reranking_time:.2f} seconds")
    
    return reranked_docs

def display_results(results: List[Tuple[Document, float]], query: str):
    """Display search results in a readable format."""
    print("\n" + "="*80)
    print(f"SEARCH QUERY: {query}")
    print("="*80)
    
    for i, (doc, score) in enumerate(results):
        print(f"\n[{i+1}] {doc.filepath} (Score: {score:.4f})")
        print("-"*80)
        
        # Display a snippet of the content (first 200 characters)
        snippet = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
        print(snippet)
    
    print("\n" + "="*80)

def main():
    # Load the corpus
    documents = load_corpus(CORPUS_DIR)
    print(f"Loaded {len(documents)} documents")
    
    # Load models
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    print(f"Loading reranker model: {RERANKER_MODEL_NAME}")
    reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
    
    # Create embeddings for all documents
    embeddings = create_embeddings(embedding_model, documents)
    
    # Interactive search loop
    while True:
        query = input("\nEnter your search query (or 'quit' to exit): ")
        if query.lower() in ('quit', 'exit', 'q'):
            break
            
        results = semantic_search(
            query, 
            documents, 
            embeddings, 
            embedding_model, 
            reranker_model
        )
        
        display_results(results, query)

if __name__ == "__main__":
    main()
