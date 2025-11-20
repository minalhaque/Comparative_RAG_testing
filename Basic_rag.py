import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import google.generativeai as genai

# Load environment
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=api_key)

print("🚀 BASIC RAG SYSTEM")
print("=" * 50)

# STEP 1: Load PDF
print("\n📄 STEP 1: Loading PDF...")
pdf_path = "TestPdf.pdf"  # Change this to your PDF name
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"✅ Loaded {len(documents)} pages")

# STEP 2: Split into chunks
print("\n✂️  STEP 2: Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Each chunk is ~500 characters
    chunk_overlap=50     # Overlap by 50 characters
)
chunks = text_splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks")

# Show example chunk
print("\n📝 Example chunk:")
print(chunks[0].page_content[:200] + "...")

# STEP 3: Create embeddings
print("\n🧠 STEP 3: Creating embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=api_key
)
print("✅ Embedding model loaded")

# STEP 4: Store in vector database
print("\n💾 STEP 4: Storing in ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"  # Saves to disk
)
print("✅ Vector store created!")

# STEP 5: Ask a question
print("\n" + "=" * 50)
question = input("❓ Ask a question about your document: ")

print(f"\n🔍 Searching for relevant chunks...")
# Find top 3 most similar chunks
relevant_docs = vectorstore.similarity_search(question, k=3)

print(f"✅ Found {len(relevant_docs)} relevant chunks:\n")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Chunk {i}:")
    print(doc.page_content[:200] + "...\n")

# STEP 6: Generate answer with LLM
print("🤖 Generating answer with Gemini...")

# Combine chunks into context
context = "\n\n".join([doc.page_content for doc in relevant_docs])

# Create prompt
prompt = f"""Based on the following context from a document, answer the question.
If the answer is not in the context, say "I cannot find this information in the document."

Context:
{context}

Question: {question}

Answer:"""

# Generate response
model = genai.GenerativeModel('models/gemini-2.5-flash')
response = model.generate_content(prompt)

print("\n" + "=" * 50)
print("💬 ANSWER:")
print(response.text)
print("=" * 50)