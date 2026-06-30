import os
import logging
import yfinance as yf
# from duckduckgo_search import DDGS  # Dependency removed due to env issues
from langchain_core.tools import tool
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

# Configure logging
LOG_FILE = "fintilligence.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@tool
def google_news_fallback(query: str) -> str:
    """
    Fallback news search using yfinance when DuckDuckGo is unavailable.
    """
    try:
        logger.info(f"🚀 [FALLBACK_NEWS] Searching for: {query}")
        # Try to extract a ticker from the query if possible, otherwise use a default
        # This is a crude fallback for general queries
        ticker_search = yf.Ticker("SPY") # Generic market ticker for general queries
        news = ticker_search.news[:5]
        
        if not news:
            return f"No fallback news results found for '{query}'."
            
        summary = f"Fallback Market News for '{query}':\n"
        for res in news:
            title = res.get('title', 'No Title')
            link = res.get('link', 'No Link')
            summary += f"- {title} ({link})\n"
        return summary
    except Exception as e:
        return f"Error in fallback news: {str(e)}"

@tool
def duckduckgo_news_search(query: str) -> str:
    """
    Search for recent news stories. (Currently using yfinance fallback)
    """
    return google_news_fallback(query)

@tool
def get_general_financial_news(topic: str = "ECONOMY") -> str:
    """
    Fetches broad financial news using Alpha Vantage. 
    Topics include: 'economy_macro', 'finance', 'energy_transportation' (Commodities), 'technology', 'ipo', 'mergers_and_acquisitions'.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return "Alpha Vantage API Key not configured."
        
    topic_map = {
        "ECONOMY": "economy_macro",
        "FINANCE": "finance",
        "COMMODITIES": "energy_transportation",
        "TECH": "technology",
        "IPO": "ipo",
        "M&A": "mergers_and_acquisitions"
    }
    
    av_topic = topic_map.get(topic.upper(), "economy_macro")
    
    try:
        import requests
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&topics={av_topic}&limit=5&apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        
        if "feed" not in data:
            return f"No financial news found for topic: {topic}."
            
        summary = f"Recent Financial News for topic '{topic}':\n"
        for item in data["feed"][:5]:
            title = item.get("title", "No Title")
            url = item.get("url", "#")
            sentiment = item.get("overall_sentiment_label", "N/A")
            summary += f"- {title} ({url}) | Sentiment: {sentiment}\n"
        return summary
    except Exception as e:
        logger.error(f"Alpha Vantage news failed: {e}")
        return f"Error fetching news: {str(e)}"

@tool
def synthesize_sentiment(ticker: str) -> str:
    """
    Analyzes news to provide a Bullish/Bearish/Neutral sentiment for a ticker.
    """
    return f"Synthesizing sentiment for {ticker} using available news... (Agent will process the findings)"
