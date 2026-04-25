from enum import Enum
from typing import Optional

from crewai import Agent, Task


class TaskType(Enum):
    """Enumeration of available task types."""

    # Main analysis tasks
    RESEARCH = "research"
    TECHNICAL_ANALYSIS = "technical_analysis"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    REPORTING = "reporting"

    # Debate and synthesis tasks
    SCEPTIC = "sceptic"
    TRUST = "trust"
    SYNTHESIS = "synthesis"

    # Tasks enhanced with Context Storage, RAG usage
    CS_RESEARCH = "cs_research"
    CS_TECHNICAL = "cs_technical"
    CS_FUNDAMENTAL = "cs_fundamental"
    CS_REPORTING = "cs_reporting"
    CS_SCEPTIC = "cs_sceptic"
    CS_TRUST = "cs_trust"
    CS_SYNTHESIS = "cs_synthesis"

    # Concurrent architecture - Round 2 cross-awareness refine tasks
    CONCURRENT_RESEARCH_REFINE = "concurrent_research_refine"
    CONCURRENT_TECHNICAL_REFINE = "concurrent_technical_refine"
    CONCURRENT_FUNDAMENTAL_REFINE = "concurrent_fundamental_refine"


# Task configurations
TASK_CONFIGS = {
    TaskType.RESEARCH: {
        "description": (
            "Gather and analyze qualitative data and public sentiment for '{stock_symbol}' "
            "by processing Reddit discussions (using analyse_reddit_tool), "
            "at least 20 Yahoo News articles (using fetch_yahoo_news_tool), "
            "and Yahoo financial analyses (using fetch_yahoo_analysis_tool). "
            "Focus on identifying the *drivers* of sentiment and the *impact* of news. "
            "Synthesize this information to build a comprehensive picture of the current "
            "sentiment and news landscape surrounding the stock."
        ),
        "expected_output": (
            "A textual report summarizing the current public sentiment, relevant news, "
            "and analyst opinions surrounding '{stock_symbol}'. This report **must**:\n"
            "1.  Provide a concise overview of the general sentiment (positive, negative, mixed) and **highlight the key themes or topics driving this sentiment** based on Reddit analysis.\n"
            "2.  Summarize the **most impactful recent news articles** from Yahoo Finance, explaining their potential implications for the stock.\n"
            "3.  Briefly discuss the consensus from Yahoo financial analyses (e.g., earnings estimates, analyst ratings) and **point out any notable shifts or strong opinions**.\n"
            "4.  Conclude with a **short, insightful synthesis** of these findings, focusing on the overall sentiment narrative.\n"
            "The report should be purely textual and focus on insights and highlights, not raw data dumps."
        ),
    },
    TaskType.TECHNICAL_ANALYSIS: {
        "description": (
            "Perform an in-depth technical analysis of '{stock_symbol}' by fetching "
            "historical market data and calculating a wide array of technical indicators "
            "(using analyse_technical_indicators_tool). Your primary goal is to **interpret these indicators** "
            "to identify trends, patterns, support/resistance levels, and potential trading signals, "
            "explaining their significance."
        ),
        "expected_output": (
            "A detailed textual technical analysis report for '{stock_symbol}'. This report **must**:\n"
            "1.  Begin with an overview of the current price trend (e.g., uptrend, downtrend, consolidation) and its strength.\n"
            "2.  **Interpret the signals from key technical indicators** (like SMAs, EMAs, MACD, RSI, Bollinger Bands, Stochastics). For each indicator, explain what its current state suggests for the stock (e.g., 'RSI at 75 indicates overbought conditions, suggesting a potential pullback.').\n"
            "3.  Identify and describe any significant chart patterns observed (e.g., head and shoulders, double top/bottom) and their implications.\n"
            "4.  Clearly state key support and resistance levels and their importance.\n"
            "5.  Highlight any notable bullish or bearish signals or divergences, explaining the reasoning.\n"
            "6.  Conclude with a **summary of the overall technical outlook** for the stock (e.g., bullish, bearish, neutral with key levels to watch).\n"
            "The report should be purely textual and focus on insights and highlights, not raw data dumps."
        ),
    },
    TaskType.FUNDAMENTAL_ANALYSIS: {
        "description": (
            "Conduct a comprehensive fundamental analysis of '{stock_symbol}' by fetching "
            "and analyzing its financial statements (income, balance sheet, cash flow), "
            "earnings reports, and company overview from Yahoo Finance (using analyse_fundamentals_tool). "
            "Your focus is to **interpret this data** to assess its financial health, profitability, "
            "growth prospects, valuation, and overall intrinsic value, highlighting key strengths and weaknesses."
        ),
        "expected_output": (
            "A detailed textual fundamental analysis report for '{stock_symbol}'. This report **must**:\n"
            "1.  Provide a summary of the company's business model and its current market position based on the overview.\n"
            "2.  Analyze and **interpret key financial metrics and ratios** (e.g., P/E, P/S, Debt-to-Equity, ROE, profit margins) and compare them to historical values or industry peers if possible (even if qualitatively).\n"
            "3.  Discuss trends in revenue, earnings, and cash flow, **highlighting growth drivers or areas of concern**.\n"
            "4.  Assess the company's financial health, focusing on liquidity, solvency, and profitability.\n"
            "5.  Provide an **assessment of the stock's valuation** (e.g., appearing overvalued, undervalued, or fairly valued based on the analysis) and explain the reasoning.\n"
            "6.  Conclude with a **summary of the key fundamental strengths and weaknesses** and the overall fundamental outlook for the company.\n"
            "The report should be purely textual and focus on insights and highlights, not raw data dumps."
        ),
    },
    TaskType.REPORTING: {
        "description": (
            "Synthesize the sentiment analysis, technical analysis, and fundamental analysis "
            "for '{stock_symbol}', drawing from the outputs of the Stock Sentiment Agent, "
            "Technical Analyst, and Fundamental Analyst. Your goal is to formulate a cohesive, "
            "insightful, and definitive investment report. **Do not simply concatenate previous reports.** "
            "Instead, integrate these diverse insights, critically evaluate them, "
            "highlight the most crucial findings and any convergences or divergences, "
            "discuss potential risks and rewards, and provide a clear, actionable investment thesis."
        ),
        "expected_output": (
            "A comprehensive and well-structured textual investment report for '{stock_symbol}'. "
            "This report **must**:\n"
            "1.  Begin with a concise **Executive Summary** that states the overall investment thesis (e.g., Buy, Sell, Hold with price targets if applicable) and the key reasons supporting it.\n"
            "2.  **Synthesize and critically evaluate** key findings from the sentiment analysis. Highlight how public perception and news flow might impact the stock, going beyond just stating the sentiment.\n"
            "3.  **Synthesize and critically evaluate** key findings from the technical analysis. Explain how the technical outlook aligns or contrasts with other analyses and what key levels are critical.\n"
            "4.  **Synthesize and critically evaluate** key findings from the fundamental analysis. Discuss the core financial health and valuation in the context of the overall investment thesis.\n"
            "5.  **Identify and discuss convergences or divergences** between the sentiment, technical, and fundamental analyses. For example, 'While fundamentals appear strong, technical indicators suggest short-term caution.'\n"
            "6.  Clearly outline the **primary catalysts and risk factors** for the stock.\n"
            "7.  Conclude with a **well-reasoned investment outlook and a specific recommendation**, reiterating the main supporting points. The recommendation should be actionable.\n"
            "This report should be a narrative, not just a list of points. Strive for clarity, conciseness, and actionable insights."
        ),
    },
    TaskType.SCEPTIC: {
        "description": (
            "Play devil's advocate for '{stock_symbol}' investment thesis. "
            "Identify risks, weaknesses, potential downsides, and challenges to the bullish case. "
            "Challenge assumptions made by other analysts. Be critical and thorough."
        ),
        "expected_output": (
            "Critical analysis highlighting risks, weaknesses, potential problems, and downsides "
            "for '{stock_symbol}'. Focus on what could go wrong and alternative bearish scenarios."
        ),
    },
    TaskType.TRUST: {
        "description": (
            "Verify data credibility and validate claims about '{stock_symbol}'. "
            "Cross-check information across sentiment, technical, and fundamental analyses. "
            "Confirm source reliability and data accuracy."
        ),
        "expected_output": (
            "Data verification report confirming accuracy and reliability of sources used "
            "in the analysis of '{stock_symbol}'."
        ),
    },
    TaskType.SYNTHESIS: {
        "description": (
            "You are the investment committee leader. Synthesize all perspectives on '{stock_symbol}' "
            "from Researcher, Technical Analyst, Fundamental Analyst, Sceptic, and Trust Specialist. "
            "Integrate their insights, resolve conflicts, address sceptic concerns, and create a final consensus report. "
            "Provide clear investment recommendation (Buy/Sell/Hold) with price target."
        ),
        "expected_output": (
            "Comprehensive investment report including executive summary, sentiment analysis, technical analysis, "
            "fundamental analysis, risk assessment, convergences/divergences, catalysts, and final recommendation."
        ),
    },
    TaskType.CS_SCEPTIC: {
        "description": (
            "Act as devil's advocate for '{stock_symbol}' in an ongoing, multi-round debate. "
            "1. FIRST, 'Read Current Context' to review the latest entries, paying special attention to recent claims made by the Trust Builder or Analysts.\n"
            "2. Use `query_session_evidence` to search the vector database for raw source texts to scrutinize.\n"
            "3. Formulate highly targeted counter-arguments to the specific claims you just read.\n"
            "4. Use 'Add Claim to Context' with `agent_name` 'Devil's Advocate - Sceptic'.\n"
            "5. CRITICAL: You MUST link your counter-argument to the specific entry you are attacking by putting its ID in the `refutes_id` field. "
            "Example: claims=[{{'content': 'Trust agent is wrong, debt is actually short-term', 'refutes_id': 'claim_123'}}].\n"
            "Limit to MAX 1 or 2 claims per turn to allow for a dynamic back-and-forth debate. DO NOT put IDs in the content text."
        ),
        "expected_output": "SUCCESS: Sceptic debate turn completed and saved to Context Storage.",
    },
    TaskType.CS_TRUST: {
        "description": (
            "Verify data for '{stock_symbol}' in an ongoing, multi-round debate. "
            "1. FIRST, 'Read Current Context' to find the latest entries, especially the recent attacks made by the Sceptic.\n"
            "2. Use `query_session_evidence` to pull raw source text to cross-check the Sceptic's claims or the original facts.\n"
            "3. Formulate verifications or corrections based on the evidence.\n"
            "4. Use 'Add Claim to Context' with `agent_name` 'Data Verification Specialist - Trust Builder'.\n"
            "5. CRITICAL: You MUST link your verification/correction to the specific entry you are addressing using the `refutes_id` field INSIDE the claim object. "
            "Example: claims=[{{'content': 'Sceptic concern is invalid; Q3 report shows high liquidity', 'refutes_id': 'claim_456'}}].\n"
            "Limit to MAX 1 or 2 claims per turn to allow for a dynamic back-and-forth debate. DO NOT put IDs in the content text."
        ),
        "expected_output": "SUCCESS: Trust debate turn completed and saved to Context Storage.",
    },
    TaskType.CS_SYNTHESIS: {
        "description": (
            "Orchestrate the '{stock_symbol}' analysis process: "
            "1. Direct Researcher, Technical, and Fundamental analysts to populate Context Storage. "
            "2. Once done, direct Sceptic to challenge assumptions and Trust to verify data integrity. "
            "3. Ensure all agents contributed. Task is complete when this full cycle (analysis -> critique -> verification) is documented in storage."
        ),
        "expected_output": "Orchestration complete. All data verified in Context Storage.",
    },
    TaskType.CS_RESEARCH: {
        "description": (
            "Analyze sentiment for '{stock_symbol}'. "
            "Tools: `analyse_finnhub_sentiment`, `analyse_alphavantage_sentiment`, `fetch_yahoo_news`, `fetch_yahoo_analysis`. "
            "CRITICAL: Batch results. Use 'Add Fact to Context' for data and 'Add Claim to Context' for insights. "
            "Set `agent_name` to 'Senior Stock Market Researcher'. "
            "AFTER saving to Context Storage, extract the generated IDs from the success message and use `store_session_evidence` to save the raw article texts/data as proof for each ID. "
            "MAX 5 facts, 3 claims. Ultra-concise style. DO NOT summarize in final answer."
        ),
        "expected_output": "SUCCESS: Sentiment and research data gathered and saved to Context Storage.",
    },
    TaskType.CS_TECHNICAL: {
        "description": (
            "Technical analysis for '{stock_symbol}' using `analyse_technical_indicators`. "
            "Interpret trends and signals. Batch results. "
            "Set `agent_name` to 'Expert Technical Analyst'. "
            "AFTER saving to Context Storage, extract the generated IDs from the success message and use `store_session_evidence` to save the raw indicator data as proof for each ID. "
            "MAX 5 facts, 3 claims. Ultra-concise style. DO NOT summarize in final answer."
        ),
        "expected_output": "SUCCESS: Technical analysis completed and saved to Context Storage.",
    },
    TaskType.CS_FUNDAMENTAL: {
        "description": (
            "Fundamental analysis for '{stock_symbol}' using `analyse_fundamentals`. "
            "Assess health and valuation. Batch results. "
            "Set `agent_name` to 'Senior Fundamental Analyst'. "
            "AFTER saving to Context Storage, extract the generated IDs from the success message and use `store_session_evidence` to save the raw financial statements/data as proof for each ID. "
            "MAX 5 facts, 3 claims. Ultra-concise style. DO NOT summarize in final answer."
        ),
        "expected_output": "SUCCESS: Fundamental analysis completed and saved to Context Storage.",
    },
    TaskType.CS_REPORTING: {
        "description": (
            "Synthesize all data for '{stock_symbol}' from Context Storage into a definitive investment report. "
            "CRITICAL RULES:\n"
            "1. DO NOT include raw IDs (like 'fact_123' or 'claim_456') inside the report narrative. Use them only internally to resolve conflicts.\n"
            "2. Cite sources by their professional names (e.g., 'Yahoo Finance', 'Analyst Consensus', 'Technical Indicators') instead of JSON IDs.\n"
            "3. Actively resolve conflicts: if a Sceptic refuted a claim (via refutes_id), you must explain why you chose one side or how you balanced the risk.\n"
            "4. Use `query_session_evidence` to retrieve raw source texts from the vector database if you need deeper context to resolve conflicts or write the narrative.\n"
            "5. Ensure the report is a professional narrative, not a list of data points."
        ),
        "expected_output": (
            "A professional Markdown (##) report for '{stock_symbol}' containing EXACTLY these sections:\n\n"
            "## 1. Executive Summary\n"
            "Thesis (Buy/Sell/Hold), Price Target, Time Horizon, and top 3 reasons.\n\n"
            "## 2. Market Sentiment & News Landscape\n"
            "- **Sentiment Drivers**: What is currently moving the stock's social/market perception.\n"
            "- **Impactful News**: Summary of the most recent critical news articles and their implications.\n"
            "- **Analyst Consensus**: Detailed view of ratings, earnings estimates, and price targets.\n\n"
            "## 3. Technical Analysis\n"
            "Detailed interpretation of trends, specific indicator alignments (RSI, MACD, SMA), and key support/resistance levels.\n\n"
            "## 4. Fundamental Analysis\n"
            "Assessment of financial health (margins, ROE, debt), valuation ratios (P/E, PEG), and growth prospects.\n\n"
            "## 5. Convergences & Divergences\n"
            "Explicit resolution of conflicts. Address where technicals and fundamentals disagree and how Sceptic/Trust flags were handled.\n\n"
            "## 6. Risk Assessment & Catalysts\n"
            "Detailed risks identified by the Sceptic and primary upcoming drivers (catalysts).\n\n"
            "## 7. Final Recommendation\n"
            "Actionable, well-reasoned conclusion with a professional investment outlook."
        ),
    },
    TaskType.CONCURRENT_RESEARCH_REFINE: {
        "description": (
            "Cross-awareness refine for '{stock_symbol}'. Round 1 specialists (yourself, Technical, Fundamental) "
            "have already populated the Context Storage with their facts.\n"
            "1. Use 'Read Current Context' to inspect what the Technical Analyst and Fundamental Analyst recorded.\n"
            "2. Identify points where their findings strengthen, contradict, or reframe your sentiment/news view.\n"
            "3. Use 'Add Claim to Context' with `agent_name` 'Senior Stock Market Researcher'. "
            "When you correct or qualify someone else's fact/claim, set `refutes_id` to that entry's ID.\n"
            "CONSTRAINTS: MAX 2 claims, 0 new facts. DO NOT call data tools again - work only on what is in Context Storage. "
            "DO NOT put IDs in the content text."
        ),
        "expected_output": "SUCCESS: Researcher cross-awareness claims saved to Context Storage.",
    },
    TaskType.CONCURRENT_TECHNICAL_REFINE: {
        "description": (
            "Cross-awareness refine for '{stock_symbol}'. Round 1 specialists (yourself, Researcher, Fundamental) "
            "have already populated the Context Storage with their facts.\n"
            "1. Use 'Read Current Context' to inspect what the Researcher (sentiment/news) and Fundamental Analyst recorded.\n"
            "2. From the technical perspective, judge whether their findings are confirmed by the chart "
            "(e.g. positive sentiment vs. bearish breakdown) or in conflict.\n"
            "3. Use 'Add Claim to Context' with `agent_name` 'Expert Technical Analyst'. "
            "When you correct or qualify someone else's fact/claim, set `refutes_id` to that entry's ID.\n"
            "CONSTRAINTS: MAX 2 claims, 0 new facts. DO NOT call data tools again - work only on what is in Context Storage. "
            "DO NOT put IDs in the content text."
        ),
        "expected_output": "SUCCESS: Technical cross-awareness claims saved to Context Storage.",
    },
    TaskType.CONCURRENT_FUNDAMENTAL_REFINE: {
        "description": (
            "Cross-awareness refine for '{stock_symbol}'. Round 1 specialists (yourself, Researcher, Technical) "
            "have already populated the Context Storage with their facts.\n"
            "1. Use 'Read Current Context' to inspect what the Researcher (sentiment/news) and Technical Analyst recorded.\n"
            "2. From the fundamental perspective, judge whether their findings are supported by the financials "
            "(e.g. bullish technicals vs. weak earnings) or in conflict.\n"
            "3. Use 'Add Claim to Context' with `agent_name` 'Senior Fundamental Analyst'. "
            "When you correct or qualify someone else's fact/claim, set `refutes_id` to that entry's ID.\n"
            "CONSTRAINTS: MAX 2 claims, 0 new facts. DO NOT call data tools again - work only on what is in Context Storage. "
            "DO NOT put IDs in the content text."
        ),
        "expected_output": "SUCCESS: Fundamental cross-awareness claims saved to Context Storage.",
    },
}


def create_task(
    task_type: TaskType,
    agent: Agent,
    context: Optional[list[Task]] = None,
    async_execution: bool = False,
) -> Task:
    """
    Factory function to create tasks based on type.

    Args:
        task_type: The type of task to create (TaskType enum).
        agent: The agent that will execute the task.
        context: Optional context list for tasks that require previous outputs.

    Returns:
        A configured Task instance.

    Raises:
        ValueError: If task_type is not a valid TaskType.
    """
    if not isinstance(task_type, TaskType):
        raise ValueError(f"Invalid task type: {task_type}. Must be a TaskType enum value.")

    config = TASK_CONFIGS.get(task_type)
    if not config:
        raise ValueError(f"No configuration found for task type: {task_type}")

    return Task(
        description=config["description"],
        expected_output=config["expected_output"],
        agent=agent,
        async_execution=async_execution,
        context=context if context else []
    )
