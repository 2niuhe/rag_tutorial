# Embedding and Reranker Model Demo

This demo showcases the effectiveness of embedding models and reranker models for semantic search using the sentence-transformers library. It specifically uses:

- **BGE-M3**: A powerful embedding model for initial retrieval
- **BGE-Reranker-v2**: A specialized reranker model to improve search relevance

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Run the demo script:

```bash
python embedding_reranker_demo.py
```

## How It Works

1. The script loads text documents from the `../corpus` directory
2. It generates embeddings for all documents using the BGE-M3 model
3. When you enter a search query, it:
   - Embeds your query using BGE-M3
   - Retrieves the top 20 most similar documents based on cosine similarity
   - Reranks these documents using the BGE-Reranker-v2 model
   - Displays the top 5 most relevant results after reranking

## Example Queries

Try these example queries:
- "How to create a virtual machine"
- "List all available instances"
- "How to manage security groups"
- "Show server details"

## Performance

The demo demonstrates the two-stage retrieval process:
1. **Embedding-based retrieval**: Fast but less precise
2. **Reranking**: More computationally intensive but significantly improves relevance

This approach combines the efficiency of vector search with the precision of cross-encoders.
