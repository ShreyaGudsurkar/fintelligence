import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    print("❌ No PINECONE_API_KEY found in .env")
else:
    try:
        pc = Pinecone(api_key=api_key)
        indexes = pc.list_indexes()
        if not indexes:
            print("ℹ️ No indexes found in this Pinecone account.")
        else:
            print("✅ Available Indexes:")
            for idx in indexes:
                print(f"   - {idx.name} ({idx.dimension}d, {idx.metric})")
    except Exception as e:
        print(f"❌ Error connecting to Pinecone: {e}")
