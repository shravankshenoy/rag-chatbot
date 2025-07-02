# RAG Chatbot

This chatbot allows you to upload a PDF containing text,tables and formulas and ask any question based on it.


## How it works
1. PDF Upload : User uploads a PDF in the streamlit app
2. PDF to Markdown : PDF is converted to markdown using Docling
3. Chunking + Embedding : 
    * Markdown is broken into smaller chunks
    * Each chunk converted to embedding using `all-MiniLM-L6-v2 model`
4. Vector database : Embeddings are stored in ChromaDB 
5. Querying :
    * User queries are converted to embedding
    * Most relevant chunks are found using cosine similarity
    * Relevant chunks are passed to LLM as context to generate final result


## Next Steps
- [ ] Add decorators to measure time taken by each stage
- [ ] Reduce latency for PDF to markdown conversion
- [ ] Add reranking and query expansion