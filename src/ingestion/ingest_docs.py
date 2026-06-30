import os
import glob
import requests
import json
import uuid
import time
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "finance-qa")

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing API Keys in .env")

genai.configure(api_key=GOOGLE_API_KEY)

def get_pinecone_host():
    """Gets the Pinecone index host URL via REST."""
    url = f"https://api.pinecone.io/indexes/{PINECONE_INDEX_NAME}"
    headers = {"Api-Key": PINECONE_API_KEY}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()['host']

def get_embeddings(texts):
    """Generates embeddings using Google's generative AI."""
    # Split into batches of 100 to avoid API limits
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=batch,
            task_type="retrieval_document"
        )
        all_embeddings.extend(result['embeddings'])
    return all_embeddings

def ingest_portfolio_docs():
    print("🚀 Starting Portable Ingestion Process...")
    
    data_path = "src/data"
    abs_data_path = os.path.join(os.getcwd(), data_path)
    pdf_files = glob.glob(os.path.join(abs_data_path, "*.pdf"))
    
    if not pdf_files:
        print(f"⚠️ No PDF files found in {data_path}")
        return

    print(f"📚 Found {len(pdf_files)} PDF files.")
    
    documents = []
    for file_path in pdf_files:
        print(f"   - Loading {os.path.basename(file_path)}...")
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            print(f"     ✅ Loaded {len(docs)} pages")
            documents.extend(docs)
        except Exception as e:
            print(f"     ❌ Error loading {os.path.basename(file_path)}: {e}")

    if not documents:
        print("⚠️ No documents loaded.")
        return

    # 2. Split Text
    print("\n✂️ Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   ✅ Created {len(chunks)} text chunks")

    # 3. Embed & Upload
    host = get_pinecone_host()
    upsert_url = f"https://{host}/vectors/upsert"
    headers = {
        "Api-Key": PINECONE_API_KEY,
        "Content-Type": "application/json"
    }

    batch_size = 50 # Small batches for REST reliability
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"   - Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}...")
        
        batch_texts = [t.page_content for t in batch]
        batch_embeddings = get_embeddings(batch_texts)

        pinecone_vectors = []
        for j, vec in enumerate(batch_embeddings):
            chunk = batch[j]
            pinecone_vectors.append({
                "id": str(uuid.uuid4()),
                "values": vec,
                "metadata": {
                    "text": chunk.page_content,
                    "source": os.path.basename(chunk.metadata.get('source', 'Unknown')),
                    "page": chunk.metadata.get('page', 0)
                }
            })

        payload = {
            "vectors": pinecone_vectors,
            "namespace": PINECONE_NAMESPACE
        }
        
        resp = requests.post(upsert_url, headers=headers, data=json.dumps(payload))
        resp.raise_for_status()

    print("\n✅ Ingestion Complete! Data is now in Pinecone.")

if __name__ == "__main__":
    ingest_portfolio_docs()
