import json

from crewai.tools import tool

from src.services.yahoo_news_fetcher import YahooNewsFetcher
from src.tools.qdrant_tools import qdrant_service

def get_fetch_yahoo_news_tool(save_to_qdrant: bool = False):
    """
    Tool factory. Returnes configured tool.
    """

    @tool
    def fetch_yahoo_news(stock_symbol: str, count: int = 10) -> str:
        """
        Fetches recent news articles related to the given stock symbol from Yahoo Finance.

        Args:
            stock_symbol (str): The stock ticker symbol for which to fetch news articles.
            count (int): The number of news articles to fetch.

        Returns:
            str: A JSON string containing the fetched news articles.
        """
        try:
            news_fetcher = YahooNewsFetcher(stock_symbol)
            news_articles = news_fetcher.fetch_news(count=count)

            evidence_text = (
                f"--- YAHOO NEWS FOR {stock_symbol.upper()} ---\n\n"
                f"NEWS ARTICLES:\n{str(news_articles)}"
            )
            if save_to_qdrant:
                try:
                    qdrant_service.add_evidence(
                        text=evidence_text,
                        metadata={"source": "Yahoo News", "symbol": stock_symbol}
                    )
                except Exception as q_err:
                    print(f"Warning: Failed to save to Qdrant: {q_err}")

            return json.dumps(news_articles, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)

    return fetch_yahoo_news
