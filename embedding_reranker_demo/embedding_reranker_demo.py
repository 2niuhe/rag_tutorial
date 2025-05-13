#!/usr/bin/env python3
"""
Demo script to test BGE-M3 embedding model and BGE-Reranker-v2 reranker model
using sentence-transformers library on a corpus of OpenStack Nova command documentation.
"""

import os
import glob
import time
import pickle
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.cross_encoder import CrossEncoder

# Configuration
CORPUS_DIR = "../corpus"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
TOP_K_RETRIEVAL = 8  # Number of documents to retrieve with embeddings
TOP_K_RERANKED = 3    # Number of documents to show after reranking

DEVICE='cpu'
BACKEND='openvino'

# Cache files
CACHE_DIR = "cache"
EMBEDDINGS_CACHE = os.path.join(CACHE_DIR, "embeddings.pkl")
DOCUMENTS_CACHE = os.path.join(CACHE_DIR, "documents.pkl")

class Document:
    def __init__(self, doc_id: str, content: str, filepath: str):
        self.id = doc_id
        self.content = content
        self.filepath = filepath
        
    def __str__(self):
        return f"Document(id={self.id}, filepath={self.filepath})"

def load_corpus(corpus_dir: str) -> List[Document]:
    """Load all text documents from the corpus directory."""
    # Check if documents are cached
    if os.path.exists(DOCUMENTS_CACHE):
        print(f"Loading documents from cache...")
        try:
            with open(DOCUMENTS_CACHE, 'rb') as f:
                documents = pickle.load(f)
            print(f"Loaded {len(documents)} documents from cache")
            return documents
        except Exception as e:
            print(f"Error loading from cache: {e}")
            print("Will load documents from source files")
    
    # Create cache directory if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Load documents from files
    documents = []
    files = glob.glob(os.path.join(corpus_dir, "*.txt"))
    
    print(f"Loading {len(files)} documents from {corpus_dir}...")
    for i, filepath in enumerate(tqdm(files)):
        filename = os.path.basename(filepath)
        doc_id = f"doc_{i}"
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        documents.append(Document(doc_id, content, filename))
    
    # Cache the documents
    try:
        with open(DOCUMENTS_CACHE, 'wb') as f:
            pickle.dump(documents, f)
        print(f"Cached {len(documents)} documents")
    except Exception as e:
        print(f"Error caching documents: {e}")
    
    return documents

def create_embeddings(model: SentenceTransformer, documents: List[Document]) -> np.ndarray:
    """Create embeddings for all documents in the corpus."""
    # Check if embeddings are cached
    if os.path.exists(EMBEDDINGS_CACHE):
        print(f"Loading embeddings from cache...")
        try:
            with open(EMBEDDINGS_CACHE, 'rb') as f:
                embeddings = pickle.load(f)
            print(f"Loaded embeddings from cache: {embeddings.shape}")
            return embeddings
        except Exception as e:
            print(f"Error loading embeddings from cache: {e}")
            print("Will create embeddings from scratch")
    
    # Create cache directory if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Create embeddings
    print(f"Creating embeddings using {model.get_sentence_embedding_dimension()}-dimensional vectors...")
    print(f"This may take a while for the first run...")
    
    contents = [doc.content for doc in documents]
    
    # Process in smaller batches to show more granular progress
    batch_size = 16
    embeddings_list = []
    
    for i in tqdm(range(0, len(contents), batch_size), desc="Batches"):
        batch = contents[i:i+batch_size]
        batch_embeddings = model.encode(batch, convert_to_numpy=True)
        embeddings_list.append(batch_embeddings)
    
    embeddings = np.vstack(embeddings_list)
    
    # Cache the embeddings
    try:
        with open(EMBEDDINGS_CACHE, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Cached embeddings of shape {embeddings.shape}")
    except Exception as e:
        print(f"Error caching embeddings: {e}")
    
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
    print("Starting the embedding and reranker demo...")
    print("Note: First run will take longer as models are downloaded and cached")
    
    # Load the corpus
    documents = load_corpus(CORPUS_DIR)
    print(f"Loaded {len(documents)} documents")
    
    # Load models with feedback
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    print("This may take a while if it's the first time loading the model...")
    start_time = time.time()
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE, backend=BACKEND)
    print(f"Embedding model loaded in {time.time() - start_time:.2f} seconds")
    
    print(f"Loading reranker model: {RERANKER_MODEL_NAME}")
    print("This may take a while if it's the first time loading the model...")
    start_time = time.time()
    reranker_model = CrossEncoder(RERANKER_MODEL_NAME, device=DEVICE, backend=BACKEND)
    print(f"Reranker model loaded in {time.time() - start_time:.2f} seconds")
    # Create embeddings for all documents
    embeddings = create_embeddings(embedding_model, documents)
    
    print("\nSetup complete! Ready for interactive search.")
    print("Tip: Subsequent runs will be much faster due to caching.")
    
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
