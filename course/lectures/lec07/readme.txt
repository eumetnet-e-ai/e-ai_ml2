Slide 1 — Motivation: Why RAG?
- Limits of standalone LLMs
- Hallucinations, outdated knowledge
- Need for domain-specific grounding

Slide 2 — Core Idea of RAG
- Retrieve relevant documents
- Augment prompt with context
- Generate answer conditioned on sources

Slide 3 — RAG Architecture Overview
- User query
- Vector database
- Retriever
- LLM generator
- (Use RAG.png)

Slide 4 — What Is a Vector Database?
- Text → embeddings
- Similarity search instead of keywords
- High-dimensional semantic space

Slide 5 — Documents as Knowledge Base
- Code repositories
- PDFs, text files, scripts
- Folder-based document trees

Slide 6 — Document Ingestion Pipeline
- Recursive folder traversal
- File type filtering
- Text extraction (PDF vs text)

Slide 7 — Example: Loading ICON Codebase
- ICON as realistic large code example
- Thousands of heterogeneous files
- Motivation for automated search

Slide 8 — From Text to Embeddings
- Sentence transformers
- Semantic representation
- Fixed-size vectors per document

Slide 9 — Embedding Model Choice
- all-MiniLM-L6-v2
- Trade-off: speed vs quality
- 384-dimensional embeddings

Slide 10 — Embedding Results (Sanity Check)
- Number of documents
- Embedding shape
- Example content snippet

Slide 11 — Similarity Search Concept
- Distance in embedding space
- Cosine / L2 distance
- Nearest neighbors = relevant context

Slide 12 — FAISS: Fast Similarity Search
- Purpose of FAISS
- Exact vs approximate search
- IndexFlatL2 as baseline

Slide 13 — Building the FAISS Index
- Add embeddings
- Memory-resident index
- Ready for queries

Slide 14 — Querying the Vector Database
- Query → embedding
- k-nearest neighbors
- Retrieve documents + metadata

Slide 15 — Example Query: ICON Cloud Cover
- Semantic match vs keyword match
- Returned files are meaningful
- Visual example of results

Slide 16 — Why Chunking Matters
- Long documents exceed context limits
- Pages / sections as units
- Better recall and precision

Slide 17 — Adding External Tutorials
- PDFs not in code repository
- Page-level chunking
- Unified search space

Slide 18 — Enriched Retrieval Results
- Code + documentation together
- More complete answers
- Reduced hallucination risk

Slide 19 — RAG with an LLM: Concept
- Retrieved text as context
- LLM acts as reasoning engine
- Answer grounded in sources

Slide 20 — RAG with OpenAI (gpt-4o-mini)
- Prompt construction
- Context injection
- File references for transparency

Slide 21 — Example Answer (ICON Question)
- Structured, technical response
- Explicit references
- Matches domain knowledge

Slide 22 — Local RAG with Ollama
- No external API
- Privacy-preserving
- Model choice: Mistral, DeepSeek

Slide 23 — Comparing Cloud vs Local Models
- Speed
- Accuracy
- Cost
- Control and privacy

Slide 24 — Saving and Reusing the Vector DB
- Persist FAISS index
- Avoid recomputation
- Practical workflow benefit

Slide 25 — End-to-End RAG Pipeline
- Ingestion → embeddings → search → LLM
- Optional web search extension
- RAG as foundation for AI assistants
