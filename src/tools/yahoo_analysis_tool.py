import json

from crewai.tools import tool

from src.services.yahoo_analysis_fetcher import YahooAnalysisFetcher
from src.tools.qdrant_tools import qdrant_service


def get_fetch_yahoo_analysis_tool(save_to_qdrant: bool = False):
    """
    Tool factory. Returnes configured tool.
    """

    @tool
    def fetch_yahoo_analysis(ticker: str) -> str:
        """
        Fetches various analyses related to the stock ticker. The analyses include:
        - Earnings Estimate
        - Revenue Estimate
        - Growth Estimates
        - Earnings History
        - EPS Trend

        Args:
            ticker (str): The stock ticker symbol to analyze.

        Returns:
            str: A JSON string containing different types of analyses.
        """
        try:
            fetcher = YahooAnalysisFetcher(ticker)
            analysis = fetcher.fetch_analysis()

            evidence_text = (
                f"--- YAHOO FINANCIAL ANALYSIS FOR {ticker.upper()} ---\n\n"
                f"ANALYSIS DATA:\n{str(analysis)}"
            )
            if save_to_qdrant:
                try:
                    qdrant_service.add_evidence(
                        text=evidence_text,
                        metadata={"source": "Yahoo Analysis", "symbol": ticker}
                    )
                except Exception as q_err:
                    print(f"Warning: Failed to save to Qdrant: {q_err}")

            return json.dumps(analysis, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)

    return fetch_yahoo_analysis
