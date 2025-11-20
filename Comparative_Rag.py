import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import google.generativeai as genai
from rank_bm25 import BM25Okapi
import numpy as np

# Load environment
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=api_key)

print("Comparing different RAG systems")
print("=" * 70)

# Load and prepare document
print("\n Loading PDF...")
pdf_path = "TestPdf.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

print("\n Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f" Created {len(chunks)} chunks")

# Setup vector store (for semantic search)
print("\n Setting up semantic search...")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=api_key
)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# Setup BM25 (for keyword search)
print(" Setting up keyword search...")
chunk_texts = [chunk.page_content for chunk in chunks]
tokenized_chunks = [text.lower().split() for text in chunk_texts]
bm25 = BM25Okapi(tokenized_chunks)

print("\n Setup complete \n")

# Ask question
print("=" * 70)
question = input(" Ask a question: ")
print()

# METHOD 1: Semantic Search (Vector Similarity)
print(" METHOD 1: SEMANTIC SEARCH (Vector Embeddings)")
print("-" * 70)
semantic_docs = vectorstore.similarity_search(question, k=3)
print("Retrieved chunks:")
for i, doc in enumerate(semantic_docs, 1):
    print(f"\n  [{i}] {doc.page_content[:150]}...")

# METHOD 2: Keyword Search (BM25)
print("\n\n METHOD 2: KEYWORD SEARCH (BM25)")
print("-" * 70)
tokenized_query = question.lower().split()
bm25_scores = bm25.get_scores(tokenized_query)
top_indices = np.argsort(bm25_scores)[-3:][::-1]
bm25_docs = [chunks[i] for i in top_indices]
print("Retrieved chunks:")
for i, doc in enumerate(bm25_docs, 1):
    print(f"\n  [{i}] {doc.page_content[:150]}...")

# METHOD 3: Hybrid Search (Combine both)
print("\n\n METHOD 3: HYBRID SEARCH (Semantic + Keyword)")
print("-" * 70)
# Get more candidates from semantic search
semantic_candidates = vectorstore.similarity_search(question, k=10)
# Re-rank using BM25 scores
hybrid_scores = []
for doc in semantic_candidates:
    idx = chunk_texts.index(doc.page_content)
    hybrid_scores.append((doc, bm25_scores[idx]))
# Sort and get top 3
hybrid_scores.sort(key=lambda x: x[1], reverse=True)
hybrid_docs = [doc for doc, score in hybrid_scores[:3]]
print("Retrieved chunks:")
for i, doc in enumerate(hybrid_docs, 1):
    print(f"\n  [{i}] {doc.page_content[:150]}...")

# Generate answers with each method
print("\n\n" + "=" * 70)
print(" COMPARING ANSWERS")
print("=" * 70)

model = genai.GenerativeModel('models/gemini-2.5-flash')

methods = [
    ("SEMANTIC SEARCH", semantic_docs),
    ("KEYWORD SEARCH", bm25_docs),
    ("HYBRID SEARCH", hybrid_docs)
]

for method_name, docs in methods:
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""Based on the following context, answer the question briefly (2-3 sentences).

Context:
{context}

Question: {question}

Answer:"""
    
    response = model.generate_content(prompt)
    
    print(f"\n{method_name}:")
    print(response.text)
    print("-" * 70)