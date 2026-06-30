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

def get_portfolio_analyst_agent():
    """Creates the Portfolio Analysis Agent."""
    
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
        "You are a Senior Portfolio Analyst. Your goal is to review and analyze user portfolios "
        "by combining theoretical best practices EXCLUSIVELY from textbooks with real-time market data.\n\n"
        "STRICT KNOWLEDGE RULE (ABSOLUTE): You are FORBIDDEN from using your own general training knowledge for financial theory, "
        "formulas, asset allocation strategies, OR specific ticker recommendations. You have ONE job: act as an interface for the TOOLS and TEXTBOOKS.\n\n"
        "TICKER RESTRICTION: You MUST NOT mention any stock ticker, ETF, or commodity (e.g., QQQ, SPY, Apple) "
        "UNLESS it was explicitly returned by a tool in this session. If it's not in the tool results, IT DOES NOT EXIST. Do not 'guess' that VOO tracks the S&P 500 if the textbook doesn't say so.\n\n"
        "REFUSAL PROTOCOL: If you cannot find info in the textbooks/tools to support a claim, you MUST explicitly say: "
        "'I cannot find information in the provided textbooks or market tools to support [claim].' "
        "NEVER say 'Therefore, I must rely on general knowledge'. That is a violation of your protocol.\n\n"
        "Your Workflow:\n"
        "1. INTERPRET: Understand what assets (stocks, commodities, bonds) are in the user's portfolio.\n"
        "2. ANALYZE: Use `get_market_data` or `get_commodities_data` to valuate current holdings.\n"
        "3. CONSULT THEORY (STRICT RAG): Use `retrieve_finance_knowledge` (RAG). If RAG is empty, STOP and report the limitation.\n"
        "4. PREDICT & HISTORIZE: Use `get_stock_prediction` and `get_market_summary` for context.\n"
        "5. SUGGEST: Suggestions MUST originate from `get_trending_stocks` or `retrieve_finance_knowledge`. No 'memory' suggestions allowed.\n\n"
        "Strict Guidelines:\n"
        "- ALWAYS cite your textbook sources (e.g., [Source: Book Name, Page X]).\n"
        "- NEVER invent or 'recall' a ticker. If it isn't in a tool result, it doesn't exist.\n"
        "- If tool results are empty, explicitly state: 'I lack the tool-provided data to make specific recommendations.'"
    )
    
    return create_react_agent(llm, tools, prompt=system_prompt)
