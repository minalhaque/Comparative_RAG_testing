# Comparative_RAG_testing
A Retrieval-Augmented Generation (RAG) system that empirically demonstrates the performance differences between semantic search, keyword search, and hybrid retrieval strategies.

# Comparative RAG System: Empirical Analysis of Retrieval Methods

A Retrieval-Augmented Generation (RAG) system that empirically demonstrates the performance differences between semantic search, keyword search, and hybrid retrieval strategies.

## Motivation

Most RAG implementations default to a single retrieval method—typically vector similarity search. However, this approach may not be optimal for all query types. This project systematically compares three retrieval strategies to quantify their trade-offs and identify when each method excels.

## Key Findings

### Empirical Testing Methodology
- **Test Corpus**: Technical ML research paper (30 pages, 249 chunks)
- **Query Set**: 8 queries spanning conceptual, technical, and mixed question types
- **Evaluation Metric**: Chunk overlap analysis between retrieval methods

### Main Results

**Finding 1: Minimal Retrieval Overlap**
- Average chunk overlap between Semantic and BM25: **0.1/3 chunks (3.3%)**
- In 7 out of 8 queries, methods retrieved **completely different information**
- **Implication**: Relying on a single retrieval method results in substantial information loss

**Finding 2: Method-Specific Strengths**

| Query Type | Example | Best Method | Reason |
|------------|---------|-------------|---------|
| Conceptual | "What is the main idea?" | Semantic Search | Captures semantic relationships |
| Technical Terms | "What is NP-hard?" | Keyword Search | Exact term matching |
| Mixed | "What algorithms are used?" | Hybrid Search | Balances both approaches |

**Finding 3: Production Implications**
- No single method dominates across all query types
- Hybrid search provides most consistent performance
- Query classification could route to optimal method per query type

## System Architecture
```
┌─────────────────┐
│   PDF Document  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Text Splitting (500 chars/50   │
│ overlap) → 249 chunks           │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         Parallel Retrieval Paths        │
│  ┌──────────────────────────────────┐   │
│  │  1. Semantic Search              │   │
│  │     - Google text-embedding-004  │   │
│  │     - ChromaDB vector store      │   │
│  │     - Cosine similarity          │   │
│  └──────────────────────────────────┘   │
│  ┌──────────────────────────────────┐   │
│  │  2. Keyword Search (BM25)        │   │
│  │     - Term frequency analysis    │   │
│  │     - Inverse document frequency │   │
│  └──────────────────────────────────┘   │
│  ┌──────────────────────────────────┐   │
│  │  3. Hybrid Search                │   │
│  │     - Semantic candidates (k=10) │   │
│  │     - Re-ranked by BM25 scores   │   │
│  └──────────────────────────────────┘   │
└─────────┬───────────────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│  LLM Generation                 │
│  (Gemini 2.5 Flash)             │
└─────────┬───────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│  Side-by-Side Comparison UI     │
└─────────────────────────────────┘
```

## Technical Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| **Embeddings** | Google text-embedding-004 | Free tier, 768-dim, competitive quality |
| **Vector Store** | ChromaDB | Local deployment, persistent storage, simple setup |
| **LLM** | Gemini 2.5 Flash | Free tier, 2-3s latency, sufficient quality |
| **Keyword Search** | BM25 (rank-bm25) | Industry standard, O(n) complexity |
| **Framework** | LangChain + Streamlit | Rapid prototyping, clean abstractions |

### Design Decisions

**Chunking Strategy**
- **Size**: 500 characters with 50-character overlap
- **Rationale**: Tested [256, 512, 1024]; 500 balanced precision vs. context
- **Trade-off**: Smaller chunks = higher precision but less context per chunk

**Embedding Model Selection**
- **Chosen**: Google text-embedding-004 (free)
- **Alternative**: OpenAI text-embedding-3-small ($0.02/1M tokens)
- **Decision**: Free tier adequate for prototype; would reevaluate for production scale

**Vector Database**
- **Chosen**: ChromaDB (local)
- **Production Alternative**: Pinecone/Weaviate for horizontal scaling
- **Trade-off**: Simplicity vs. production readiness

## Performance Metrics

### Retrieval Diversity Analysis

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Avg overlap (Semantic ∩ BM25) | 0.1/3 (3.3%) | Minimal redundancy |
| Queries with 0% overlap | 7/8 (87.5%) | High method diversity |
| Unique chunks per query | 3-5 | Broad information coverage |

### Query-Level Results
```
Query: "What is the main idea of this paper?"
├─ Semantic: Conceptual overview chunks (0% overlap with BM25)
├─ BM25: Technical detail chunks
└─ Hybrid: Balanced selection
Result: Methods retrieved completely different information

Query: "What is NP-hard?"
├─ Semantic: Related complexity theory concepts
├─ BM25: Exact "NP-hard" mentions
└─ Hybrid: Combined approach
Result: 0/3 chunks in common
```

## Installation & Usage

### Prerequisites
- Python 3.8+
- Google API key ([Get one here](https://aistudio.google.com/app/apikey))

### Setup
Clonse Repository 
git clone https://github.com/minalhaque/comparative-rag.git
cd comparative-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
echo "GOOGLE_API_KEY=your_key_here" > .env
```

### Running the System

**Command-line interface:**
python comparative_rag.py
```

**Run empirical analysis:**
python test_and_analyze.py
```

## Project Structure
```
comparative-rag/
├── Basic_rag.py              # Single-method RAG implementation
├── Comparative_Rag.py        # Three-method comparison (CLI)
├── app.py                    # Streamlit web interface
├── test_and_analyze.py       # Empirical evaluation script
├── requirements.txt          # Python dependencies
├── .env                      # API keys (not committed)
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Future Work

### Short-term Improvements
1. **Query Classification**: Automatically route queries to optimal retrieval method
2. **Cross-Encoder Reranking**: Improve precision by 10-20% with reranking model
3. **Expanded Evaluation**: Test on diverse document types (legal, medical, code)

### Production Considerations
4. **Caching Layer**: Redis for frequent queries to reduce API costs
5. **Monitoring**: Track retrieval quality metrics over time
6. **A/B Testing Framework**: Compare methods with real user queries
7. **Cost Optimization**: Batch processing, embedding caching

### Research Extensions
8. **Benchmark Evaluation**: Test on BEIR, MS MARCO datasets
9. **Learned Routing**: ML model to predict best retrieval method per query
10. **Multi-document RAG**: Extend to multiple source documents

## Lessons Learned

1. **Retrieval Quality >> Generation Quality**: Even GPT-4 cannot compensate for poor retrieval
2. **No Universal Best Method**: Performance is query-dependent
3. **Metrics are Essential**: Subjective evaluation is insufficient
4. **Hybrid Trade-offs**: Complexity increase doesn't always justify performance gains

## References & Resources

- [Retrieval-Augmented Generation Paper](https://arxiv.org/abs/2005.11401)
- [BM25 Algorithm Explanation](https://en.wikipedia.org/wiki/Okapi_BM25)
- [LangChain Documentation](https://python.langchain.com/docs/tutorials/rag/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

## Contributing

This is a learning project, but feedback and suggestions are welcome. Feel free to:
- Open an issue for bugs or feature requests
- Submit a pull request with improvements
- Share your own RAG experiments

## License

MIT License - feel free to use this code for learning and experimentation.

## Author

Minal Haque 
- GitHub: [@minalhaque](https://github.com/minalhaque)
- LinkedIn: [Minal Haque](https://www.linkedin.com/in/minalhaque)
- Email: syedaminalhaque@gmail.com

---

**Built in 1 day as a practical exploration of RAG retrieval methods.**
