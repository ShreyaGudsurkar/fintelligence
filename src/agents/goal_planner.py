from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from src.rag.retriever import get_retriever_tool
from src.utils.finance_tools import (
    calculate_compound_interest,
    generate_amortization_schedule,
    calculate_goal_feasibility
)
from src.utils.market_tools import (
    get_market_data,
    get_stock_prediction,
    get_stock_news,
    get_commodities_data,
    get_market_summary,
    get_trending_stocks
)
from dotenv import load_dotenv

load_dotenv()

def get_goal_planner_agent():
    """Creates the Goal Planning Agent."""
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
    
    # Get Tools (Full intelligence suite for math + market + RAG)
    tools = [
        get_retriever_tool(),
        get_market_data,
        get_stock_prediction,
        get_stock_news,
        get_commodities_data,
        get_market_summary,
        get_trending_stocks,
        calculate_compound_interest,
        generate_amortization_schedule,
        calculate_goal_feasibility
    ]
    
    # Define Prompt
    system_prompt = (
        "You are a Professional Financial Planner. Your goal is to help users set and achieve "
        "accurate financial goals using math-based calculators.\n\n"
        "STRICT KNOWLEDGE RULE (ABSOLUTE): You are FORBIDDEN from using your own general training knowledge for financial theory, "
        "formulas, OR specific ticker recommendations. You MUST NOT mention any asset (e.g. QQQ, SPY) unless it appears in tool results.\n\n"
        "REFUSAL PROTOCOL: If information is missing from your tools/textbooks, you MUST explicitly state the limitation. "
        "NEVER fall back to general knowledge. Saying 'I must rely on general knowledge' is a protocol failure.\n\n"
        "Your Workflow:\n"
        "1. ASK: If the user query is vague, ask for the missing variables.\n"
        "2. COLLABORATE: Incorporate 'Portfolio Analysis Reports' from this session into your goal calculations.\n"
        "3. CALCULATE: Use your financial tools for ALL projections. No mental math or guesses.\n"
        "4. CONTEXTUALIZE: Use `get_market_summary` or `get_trending_stocks` for ROI context.\n"
        "5. STRUCTURE: Present your plan with clear headers and summary tables.\n\n"
        "Strict Guidelines:\n"
        "- HYBRID PLANNING: Compare 'Goal ROI' with 'Portfolio ROI' found in conversation history. Highlight gaps.\n"
        "- NO GENERAL KNOWLEDGE: If you don't know an asset's purpose (e.g. what QQQ tracks) because RAG is empty, you MUST NOT guess. Ask the user or report it as unknown.\n"
        "- ALWAYS show your math and provide one financial disclaimer at the end."
    )
    
    return create_react_agent(llm, tools, prompt=system_prompt)
