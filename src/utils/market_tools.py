import os
import logging
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from langchain_core.tools import tool
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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

# Retry Configuration: Retry 3 times, wait 2^x * 1 second between retries
RETRY_CONFIG = {
    "stop": stop_after_attempt(3),
    "wait": wait_exponential(multiplier=1, min=2, max=10),
    "retry": retry_if_exception_type((Exception,)), # Retry on any exception for now, can be more specific
    "reraise": True
}

@retry(**RETRY_CONFIG)
def _fetch_yfinance_price(symbol: str):
    logger.info(f"Fetching yFinance price for {symbol}...")
    ticker = yf.Ticker(symbol)
    history = ticker.history(period="1d")
    if history.empty:
        raise ValueError(f"No price data found for {symbol}")
    return history['Close'].iloc[-1]

@retry(**RETRY_CONFIG)
def _fetch_alpha_vantage_price(symbol: str, api_key: str):
    logger.info(f"Fetching Alpha Vantage price for {symbol}...")
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, _ = ts.get_quote_endpoint(symbol)
    # '05. price' is the key for quote endpoint
    return float(data['05. price'].iloc[0])

@retry(**RETRY_CONFIG)
def _fetch_yfinance_market_index(symbol: str):
    logger.info(f"Fetching market index {symbol}...")
    ticker = yf.Ticker(symbol)
    history = ticker.history(period="1d")
    if history.empty:
         raise ValueError(f"No data found for index {symbol}")
    return history['Close'].iloc[-1]

@retry(**RETRY_CONFIG)
def _fetch_yfinance_news(symbol: str):
    logger.info(f"Fetching news for {symbol}...")
    ticker = yf.Ticker(symbol)
    news = ticker.news
    if not news:
        # Sometimes news list is empty, don't necessarily want to retry if it's just no news,
        # but if it's a network glitch, we do. Let's treat empty news as a valid result to avoid loops
        # or maybe raise if we suspect it's an error. For now, simply return.
        return []
    return news

@tool
def get_market_data(symbol: str) -> str:
    """
    Fetches market data for a given stock symbol including price and key fundamentals (Market Cap, P/E, etc.)
    """
    # 1. Fundamental Data via yFinance
    ticker = yf.Ticker(symbol)
    info = ticker.info
    
    price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
    market_cap = info.get('marketCap', 'N/A')
    pe_ratio = info.get('trailingPE', 'N/A')
    range_52w = f"{info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}"
    dividend_yield = info.get('dividendYield', 'N/A')
    summary = info.get('longBusinessSummary', 'No summary available.')[:500] + "..."

    # Formatting Market Cap
    if isinstance(market_cap, (int, float)):
        market_cap = f"${market_cap / 1e9:.2f}B"

    # Formatting Dividend Yield
    if isinstance(dividend_yield, (int, float)):
        dividend_yield = f"{dividend_yield * 100:.2f}%"

    data_summary = (
        f"Market Data for {symbol}:\n"
        f"- Current Price: ${price}\n"
        f"- Market Cap: {market_cap}\n"
        f"- P/E Ratio: {pe_ratio}\n"
        f"- 52-Week Range: {range_52w}\n"
        f"- Dividend Yield: {dividend_yield}\n"
        f"- Business Summary: {summary}\n"
    )
    return data_summary

@tool
def get_market_summary() -> str:
    """
    Provides a general market summary (S&P 500, Nasdaq) using yFinance.
    """
    try:
        sp500 = _fetch_yfinance_market_index("^GSPC")
        nasdaq = _fetch_yfinance_market_index("^IXIC")
        
        return (
            f"General Market Summary:\n"
            f"- S&P 500: {sp500:.2f}\n"
            f"- Nasdaq: {nasdaq:.2f}"
        )
    except Exception as e:
         logger.error(f"Failed to fetch market summary: {e}")
         return f"Error fetching market summary: {str(e)}"

@tool
def get_trending_stocks() -> str:
    """
    Fetches the top gainers, losers, and most active stocks in the US market using Alpha Vantage.
    Useful for answering 'What is trending today?' or 'Show me the top gainers.'
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return "Alpha Vantage API Key not configured."
    
    try:
        import requests
        url = f"https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        
        # Check for API limit or error messages
        if "Information" in data or "Note" in data:
            info = data.get("Information") or data.get("Note")
            logger.warning(f"Alpha Vantage info in trending stocks: {info}")
            return "Trending stocks data is currently unavailable due to API rate limits. Please try again in top of the hour."

        if not any(k in data for k in ["top_gainers", "top_losers", "most_active"]):
            return "Could not fetch trending stocks at this time. The data might not be available for the current market session."
            
        summary = "Today's Trending Stocks:\n\n"
        
        if "top_gainers" in data and data["top_gainers"]:
            summary += "**🔥 Top Gainers:**\n"
            for stock in data["top_gainers"][:5]:
                summary += f"- {stock['ticker']}: {stock['price']} ({stock['change_percentage']})\n"
        
        if "top_losers" in data and data["top_losers"]:
            summary += "\n**❄️ Top Losers:**\n"
            for stock in data["top_losers"][:5]:
                summary += f"- {stock['ticker']}: {stock['price']} ({stock['change_percentage']})\n"
            
        if "most_active" in data and data["most_active"]:
            summary += "\n**📈 Most Active:**\n"
            for stock in data["most_active"][:5]:
                summary += f"- {stock['ticker']}: {stock['price']} (Vol: {stock['volume']})\n"
            
        return summary
    except Exception as e:
        logger.error(f"Failed to fetch trending stocks: {e}")
        return "Trending stocks data is currently restricted. Try checking major financial news for broader market movements."

@tool
def get_commodities_data(commodity_type: str = "ALL") -> str:
    """
    Fetches real-time prices for major commodities like Crude Oil, Gold, Natural Gas, etc.
    `commodity_type` can be: WTI (Oil), BRENT (Oil), NATURAL_GAS, COPPER, ALUMINUM, WHEAT, CORN, COTTON, SUGAR, COFFEE, or ALL.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return "Alpha Vantage API Key not configured."
        
    # Mapping of common names to AV function names
    functions = {
        "WTI": "WTI",
        "BRENT": "BRENT",
        "NATURAL_GAS": "NATURAL_GAS",
        "COPPER": "COPPER",
        "ALUMINUM": "ALUMINUM",
        "WHEAT": "WHEAT",
        "CORN": "CORN",
        "COTTON": "COTTON",
        "SUGAR": "SUGAR",
        "COFFEE": "COFFEE",
        "GOLD": "GOLD",
        "SILVER": "SILVER"
    }
    
    import requests
    
    def _fetch_av_comm(func):
        # Try daily interval first for most recent data
        url = f"https://www.alphavantage.co/query?function={func}&interval=daily&apikey={api_key}"
        res = requests.get(url).json()
        
        # Fallback to monthly if daily fails
        if "data" not in res or not res["data"]:
            url = f"https://www.alphavantage.co/query?function={func}&interval=monthly&apikey={api_key}"
            res = requests.get(url).json()
            
        if "Information" in res or "Note" in res:
            # Return None to trigger yfinance fallback in the parent function
            return None
            
        if "data" in res and res["data"]:
            latest = res["data"][0]
            unit = res.get("unit", "")
            return f"{latest['value']} {unit} (as of {latest['date']})"
            
        return None

    def _fetch_yf_comm(commodity_type):
        ticker_map = {
            "GOLD": "GC=F",
            "SILVER": "SI=F",
            "WTI": "CL=F",
            "BRENT": "BZ=F",
            "NATURAL_GAS": "NG=F",
            "COPPER": "HG=F"
        }
        sym = ticker_map.get(commodity_type.upper())
        if not sym: return None
        try:
            ticker = yf.Ticker(sym)
            price = ticker.info.get('regularMarketPrice') or ticker.history(period="1d")['Close'].iloc[-1]
            return f"{price:.2f} USD (Source: Yahoo Finance)"
        except Exception as e:
            logger.warning(f"yFinance fallback failed for {commodity_type}: {e}")
            return None

    try:
        # 1. Try Alpha Vantage first (Primary)
        if commodity_type.upper() in functions:
            val = _fetch_av_comm(functions[commodity_type.upper()])
            if val:
                return f"Current {commodity_type} Price: {val}"
        
        # 2. Level 2 Fallback: yFinance for commodities
        yf_val = _fetch_yf_comm(commodity_type)
        if yf_val:
            return f"Current {commodity_type} Price: {yf_val}"
            
        # 3. Final Fallback: Error message
        return f"MARKET_DATA_UNAVAILABLE: Technical data limit reached for Alpha Vantage and yFinance fallback failed. Please check the general news for recent {commodity_type} updates."

    except Exception as e:
        logger.error(f"Failed to fetch commodities: {e}")
        return f"Error fetching commodities: {str(e)}"

@tool
def get_stock_news(symbol: str) -> str:
    """
    Fetches recent news for a stock symbol using yFinance, falling back to Alpha Vantage.
    """
    summary = ""
    
    # 1. Try yFinance
    try:
        news = _fetch_yfinance_news(symbol)
        if news:
            summary += f"Recent News for {symbol} (Source: Yahoo Finance):\n"
            for item in news[:3]:
                title = item.get('title')
                if not title and 'content' in item:
                    title = item['content'].get('title')
                if not title:
                    title = 'No Title'
                    
                # Try getting link from top level, or nested in clickThroughUrl
                link = item.get('link')
                if not link and 'clickThroughUrl' in item:
                    link = item['clickThroughUrl'].get('url')
                if not link:
                    link = "#"
                    
                summary += f"- {title} ({link})\n"
            return summary
    except Exception as e:
        logger.warning(f"yFinance news failed: {e}")

    # 2. Fallback: Alpha Vantage
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if api_key:
        try:
            import requests
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&limit=3&apikey={api_key}"
            response = requests.get(url)
            data = response.json()
            
            if "feed" in data:
                summary += f"Recent News for {symbol} (Source: Alpha Vantage):\n"
                for item in data["feed"][:3]:
                    title = item.get("title", "No Title")
                    url = item.get("url", "#")
                    summary += f"- {title} ({url})\n"
                return summary
        except Exception as e:
            logger.error(f"Alpha Vantage news failed: {e}")

    return f"No news found for {symbol}."

@tool
def get_stock_prediction(symbol: str) -> str:
    """
    Fetches analyst price targets and recommendations for a stock symbol using yFinance.
    Includes Mean/High/Low Target Price and Analyst Recommendation (Buy/Sell/Hold).
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        target_mean = info.get('targetMeanPrice', 'N/A')
        target_high = info.get('targetHighPrice', 'N/A')
        target_low = info.get('targetLowPrice', 'N/A')
        recommendation = info.get('recommendationKey', 'N/A').upper()
        num_analysts = info.get('numberOfAnalystOpinions', 'N/A')
        
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
        
        return (
            f"Analyst Predictions for {symbol}:\n"
            f"- Current Price: {current_price}\n"
            f"- Consensus Recommendation: {recommendation}\n"
            f"- Target Mean Price: {target_mean}\n"
            f"- Target High: {target_high}\n"
            f"- Target Low: {target_low}\n"
            f"- Number of Analysts: {num_analysts}\n"
        )
    except Exception as e:
        logger.error(f"Failed to fetch predictions for {symbol}: {e}")
        return f"Error fetching predictions for {symbol}: {str(e)}"
