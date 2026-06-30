from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from src.rag.retriever import get_retriever_tool
from dotenv import load_dotenv

load_dotenv()

# --- Agent Setup ---

def get_finance_qa_agent():
    """Creates the Finance Q&A Agent."""
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
    
    # Get Tools
    tools = [get_retriever_tool()]
    
    # Define Prompt
    system_prompt = (
        "You are an expert Financial Educator. Your goal is to explain financial concepts "
        "EXCLUSIVELY using the provided knowledge base. You are FORBIDDEN from using your own general training knowledge.\n\n"
        "REFUSAL PROTOCOL: If information is missing from your tools/textbooks, you MUST explicitly state: "
        "'I cannot find information regarding [topic] in the provided financial textbooks.' "
        "NEVER say 'Therefore, I must rely on general knowledge'. That is a protocol failure.\n\n"
        "Strict Guidelines:\n"
        "1. ALWAYS use the `retrieve_finance_knowledge` tool before answering.\n"
        "2. ONLY answer using information found in the retrieved context. Do not add outside facts.\n"
        "3. INLINE CITATIONS: For every fact or step you provide, include an inline citation (e.g., [Source: Book Name, Page 12]).\n"
        "4. MISSING INFORMATION: If the retrieved context does not contain the answer, follow the REFUSAL PROTOCOL.\n"
        "5. SOURCE LIST: At the very end of your response, list all the unique sources you cited."
    )
    
    return create_react_agent(llm, tools, prompt=system_prompt)
