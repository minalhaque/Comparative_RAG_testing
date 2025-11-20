import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import google.generativeai as genai
from rank_bm25 import BM25Okapi
import numpy as np

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=api_key)

loader = PyPDFLoader("paper.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

chunk_texts = [chunk.page_content for chunk in chunks]
tokenized_chunks = [text.lower().split() for text in chunk_texts]
bm25 = BM25Okapi(tokenized_chunks)

# Test different types of queries
test_queries = [
    # Conceptual questions (semantic should win)
    "What is the main idea of this paper?",
    "What are the theoretical implications?",
    "How does this relate to machine learning?",
    
    # Specific term questions (keyword should win)
    "What is BPE?",
    "What is NP-hard?",
    "What is tokenization?",
    
    # Mixed questions (hybrid might win)
    "What algorithms are used for tokenization?",
    "What are the computational complexity results?",
]

print(" TESTING DIFFERENT RETRIEVAL METHODS")
print("=" * 80)

results = []

for query in test_queries:
    print(f"\n📝 Query: {query}")
    print("-" * 80)
    
    # Get results from each method
    semantic_docs = vectorstore.similarity_search(query, k=3)
    
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(bm25_scores)[-3:][::-1]
    bm25_docs = [chunks[i] for i in top_indices]
    
    semantic_candidates = vectorstore.similarity_search(query, k=10)
    hybrid_scores = []
    for doc in semantic_candidates:
        idx = chunk_texts.index(doc.page_content)
        hybrid_scores.append((doc, bm25_scores[idx]))
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    hybrid_docs = [doc for doc, score in hybrid_scores[:3]]
    
    # Check which chunks each method retrieved
    semantic_chunks = [doc.page_content[:100] for doc in semantic_docs]
    bm25_chunks = [doc.page_content[:100] for doc in bm25_docs]
    hybrid_chunks = [doc.page_content[:100] for doc in hybrid_docs]
    
    # Simple metric: diversity (how many unique chunks?)
    all_chunks = set(semantic_chunks + bm25_chunks + hybrid_chunks)
    
    print(f"  Semantic retrieved: {len(set(semantic_chunks))} unique chunks")
    print(f"  BM25 retrieved: {len(set(bm25_chunks))} unique chunks")
    print(f"  Hybrid retrieved: {len(set(hybrid_chunks))} unique chunks")
    print(f"  Total unique chunks across all methods: {len(all_chunks)}")
    
    # Check overlap
    semantic_set = set(semantic_chunks)
    bm25_set = set(bm25_chunks)
    hybrid_set = set(hybrid_chunks)
    
    overlap_sem_bm25 = len(semantic_set & bm25_set)
    overlap_sem_hybrid = len(semantic_set & hybrid_set)
    overlap_bm25_hybrid = len(bm25_set & hybrid_set)
    
    print(f"  Overlap (Semantic ∩ BM25): {overlap_sem_bm25}/3")
    print(f"  Overlap (Semantic ∩ Hybrid): {overlap_sem_hybrid}/3")
    print(f"  Overlap (BM25 ∩ Hybrid): {overlap_bm25_hybrid}/3")
    
    # Determine which seems best
    if overlap_sem_bm25 == 0:
        print("  💡 Finding: Methods retrieved COMPLETELY DIFFERENT chunks!")
    elif overlap_sem_bm25 == 3:
        print("  💡 Finding: Methods retrieved IDENTICAL chunks")
    
    results.append({
        'query': query,
        'semantic_unique': len(set(semantic_chunks)),
        'bm25_unique': len(set(bm25_chunks)),
        'hybrid_unique': len(set(hybrid_chunks)),
        'total_unique': len(all_chunks),
        'overlap_sem_bm25': overlap_sem_bm25
    })

# Summary
print("\n\n" + "=" * 80)
print("📊 SUMMARY OF FINDINGS")
print("=" * 80)

avg_overlap = sum([r['overlap_sem_bm25'] for r in results]) / len(results)
print(f"\nAverage overlap between Semantic and BM25: {avg_overlap:.1f}/3 chunks")

if avg_overlap < 1:
    print("✅ Different methods retrieve SIGNIFICANTLY different information")
    print("   → This validates the need for comparing multiple approaches!")
elif avg_overlap > 2:
    print("⚠️ Methods retrieve very similar information")
    print("   → May need to adjust parameters or use more diverse queries")

# Count query types
conceptual_queries = test_queries[:3]
specific_queries = test_queries[3:6]

print("\n Query Type Analysis:")
print("  - Conceptual questions: Test if semantic search captures general ideas")
print("  - Specific term questions: Test if keyword search catches exact terms")
print("  - Mixed questions: Test if hybrid combines strengths")

print("\n Key Insight:")
print("  No single retrieval method is always best - the optimal choice")
print("  depends on the query type and information need.")