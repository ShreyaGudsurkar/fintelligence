from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

def get_retriever_tool():
    """Initializes and returns the Pinecone retriever tool."""
    
    # 1. Initialize Embeddings (Must match ingestion model)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 2. Get Config
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    namespace = os.getenv("PINECONE_NAMESPACE", "finance-qa")
    
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME not found in environment variables")

    @tool
    def retrieve_finance_knowledge(query: str) -> str:
        """
        Useful for answering questions about general finance concepts, definitions, and educational topics.
        Use this tool to look up information in the finance textbooks.
        """
        try:
            # Lazy initialize the index inside the tool call to prevent startup crash
            pc = Pinecone(api_key=api_key)
            
            # Check if index exists first to provide better error
            # (Note: list_indexes might be better but let's try direct connection catch)
            index = pc.Index(index_name)
            
            # Generate embedding for the query
            query_embedding = embeddings.embed_query(query)
            
            # Query Pinecone
            results = index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True,
                namespace=namespace
            )
            
            # Format results
            context = ""
            for match in results['matches']:
                text = match['metadata'].get('text', '')
                source = match['metadata'].get('source', 'Unknown')
                page = match['metadata'].get('page', 'N/A')
                score = match['score']
                context += f"\n--- Source: {source} | Page: {page} | Confidence: {score:.2f} ---\n{text}\n"
                
            return context if context else "No relevant information found in the knowledge base."
            
        except Exception as e:
            if "NOT_FOUND" in str(e) or "404" in str(e):
                return f"Error: The Pinecone index '{index_name}' was not found. Please ensure the index is created and data is ingested."
            return f"Error retrieving information: {str(e)}"

    return retrieve_finance_knowledge
