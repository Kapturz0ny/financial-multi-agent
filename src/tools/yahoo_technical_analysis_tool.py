import json

from crewai.tools import tool

from src.services.yahoo_technical_analyser import YahooTechnicalAnalyser
from src.tools.qdrant_tools import qdrant_service


@tool
def analyse_technical_indicators(ticker: str, period: str = "1y") -> str:
    """
    Fetches and analyses technical indicators for a given stock ticker using Yahoo Finance.
    The analysis includes various technical indicators such as:
    - SMA (Simple Moving Average)
    - EMA (Exponential Moving Average)
    - WMA (Weighted Moving Average)
    - MACD (Moving Average Convergence Divergence)
    - RSI (Relative Strength Index)
    - Stochastic Oscillator
    - Bollinger Bands
    - ATR (Average True Range)

    Args:
        ticker (str): The stock ticker symbol to analyze.
        period (str): The time period for the analysis (default is "1y").
    Returns:

        str: A JSON string containing the fetched technical indicators.
    """
    try:
        analyser = YahooTechnicalAnalyser(ticker)
        data: dict = analyser.get_technical_data(period=period)
        evidence_text = (
            f"--- YAHOO TECHNICAL INDICATORS FOR {ticker.upper()} (Period: {period}) ---\n\n"
            f"TECHNICAL DATA:\n{str(data)}"
        )
        try:
            qdrant_service.add_evidence(
                text=evidence_text,
                metadata={"source": "Yahoo Technicals", "symbol": ticker}
            )
        except Exception as q_err:
            print(f"Warning: Failed to save to Qdrant: {q_err}")
        return json.dumps(data, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)
