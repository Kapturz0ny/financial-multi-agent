import json

from crewai.tools import tool

from src.services.yahoo_fundamental_analyser import YahooFundamentalAnalyser
from src.tools.qdrant_tools import qdrant_service


def get_analyse_fundamentals_tool(save_to_qdrant: bool = False):
    """
    Tool factory. Returnes configured tool.
    """

    @tool
    def analyse_fundamentals(ticker: str) -> str:
        """
        Fetches a comprehensive profile for a given stock ticker.
        This tool retrieves a wide array of information about a publicly traded company,
        encapsulated in a single JSON object. The data includes:
            - General company details: Address, website, industry, sector, business summary, number of employees.
            - Key executives and company officers.
            - Corporate governance risk scores.
            - Current and historical stock performance metrics: Open, high, low, close, volume, market cap, beta.
            - Dividend information: Rate, yield, ex-dividend date, payout ratio.
            - Valuation metrics: Trailing P/E, forward P/E, price-to-sales, price-to-book.
            - Moving averages: 50-day and 200-day averages.
            - Share statistics: Shares outstanding, float, shares short, insider/institutional ownership.
            - Financial highlights: Total cash, total debt, total revenue, profit margins, return on assets/equity, EBITDA.
            - Earnings data: Trailing EPS, forward EPS, earnings quarterly growth.
            - Analyst recommendations: Target prices (high, low, mean, median), recommendation mean, number of opinions.
            - Information about the last stock split.
            - Market state and trading session details.

        Args:
            ticker (str): The stock ticker symbol to analyze.

        Returns:
            str: A JSON string containing the fetched profile data.
        """
        try:
            fetcher = YahooFundamentalAnalyser(ticker)
            analysis = fetcher.fetch_fundamentals()

            evidence_text = (
                f"--- YAHOO FUNDAMENTALS FOR {ticker.upper()} ---\n\n"
                f"FUNDAMENTAL DATA:\n{str(analysis)}"
            )
            if save_to_qdrant:
                try:
                    qdrant_service.add_evidence(
                        text=evidence_text,
                        metadata={"source": "Yahoo Fundamentals", "symbol": ticker}
                    )
                except Exception as q_err:
                    print(f"Warning: Failed to save to Qdrant: {q_err}")

            return json.dumps(analysis, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)

    return analyse_fundamentals
