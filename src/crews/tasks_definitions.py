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
    CS_DEBATE_ORCHESTRATION = "cs_debate_orchestartion"


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
    TaskType.CS_RESEARCH: {
        "description": (
            "Analyze sentiment and news for '{stock_symbol}' using Finnhub, AlphaVantage, and Yahoo tools. "
            "YOU MUST STRICTLY FOLLOW THIS EXACT 3-STEP WORKFLOW:\n"
            "STEP 1: Fetch news and financial analyses.\n"
            "STEP 2: Use 'add_facts_to_context' to save the hard data (at least 5 facts). Set `agent_name` to 'Senior Stock Market Researcher'.\n"
            "STEP 3: Use 'add_claims_to_context' to save your qualitative insights (at least 2 claims). Summarize impactful news and consensus.\n"
            "CRITICAL: Do not skip any steps. Do not summarize in the final answer, just confirm completion."
        ),
        "expected_output": "SUCCESS: Facts and Claims added.",
    },
    TaskType.CS_TECHNICAL: {
        "description": (
            "Perform technical analysis on '{stock_symbol}' using `analyse_technical_indicators_tool`. "
            "YOU MUST STRICTLY FOLLOW THIS EXACT 3-STEP WORKFLOW:\n"
            "STEP 1: Fetch technical indicators.\n"
            "STEP 2: Use 'add_facts_to_context' to save the raw metrics (e.g., exact RSI, SMA values). Set `agent_name` to 'Expert Technical Analyst'. At least 5 facts.\n"
            "STEP 3: Use 'add_claims_to_context' to save your interpretations (e.g., 'RSI at 70 suggests overbought'). At least 2 claims.\n"
            "CRITICAL: Do not skip any steps. Do not summarize in the final answer, just confirm completion."
        ),
        "expected_output": "SUCCESS: Facts and Claims added.",
    },
    TaskType.CS_FUNDAMENTAL: {
        "description": (
            "Conduct fundamental analysis on '{stock_symbol}' using `analyse_fundamentals_tool`. "
            "YOU MUST STRICTLY FOLLOW THIS EXACT 3-STEP WORKFLOW:\n"
            "STEP 1: Fetch financial statements and metrics.\n"
            "STEP 2: Use 'add_facts_to_context' to save the financial data (e.g., P/E, ROE). Set `agent_name` to 'Senior Fundamental Analyst'. At least 5 facts.\n"
            "STEP 3: Use 'add_claims_to_context' to save your valuation assessment (overvalued/undervalued). At least 2 claims.\n"
            "CRITICAL: Do not skip any steps. Do not summarize in the final answer, just confirm completion."
        ),
        "expected_output": "SUCCESS: Facts and Claims added.",
    },
    TaskType.CS_REPORTING: {
        "description": (
            "Synthesize all data for '{stock_symbol}' into a definitive, institutional-grade investment report. "
            "WORKFLOW:\n"
            "1. Use 'read_current_context' to load all facts and claims from the debate.\n"
            "2. If you need deeper context or raw data, use 'query_session_evidence' to search the Vector DB.\n"
            "3. Write the report following these editorial guidelines:\n\n"
            "- TONE & STYLE: Write in a highly professional, academic financial tone. Use advanced market terminology and well-structured, comprehensive, long sentences.\n"
            "- DATA DENSITY: Base your analysis on hard quantitative data. Include exact numbers, percentages (%), dollar amounts ($), dates, and specific financial metrics (e.g., P/E, ROE, RSI, MACD).\n"
            "- CITATIONS: Explicitly attribute data to its origin using phrases like 'according to', 'source:', or 'analysts estimate'.\n"
            "- OBJECTIVITY & SENTIMENT: Maintain strict analytical neutrality. You must equally evaluate the bull case (highlighting growth, strong profits, upside, momentum, robust gains, and opportunities to outperform) against the bear case (highlighting risks of decline, weak performance, volatility, downside threats, losses, and concerns).\n"
            "- CONFLICT RESOLUTION: Explicitly address the Sceptic/Trust debate from the context and explain your final judgment.\n"
            "- NO RAW IDs: Write for a human executive. Never include JSON IDs (like 'fact_123') in the final text."
        ),
        "expected_output": (
            "A comprehensive, highly actionable Markdown report for '{stock_symbol}'. You MUST use EXACTLY these 8 main section headers (##):\n\n"
            "## Executive Summary\n"
            "## Sentiment Analysis\n"
            "## Technical Analysis\n"
            "## Fundamental Analysis\n"
            "## Convergences/Divergences\n"
            "## Risk Assessment\n"
            "## Catalysts\n"
            "## Final Recommendation\n\n"
            "FORMATTING & STRUCTURAL RULES:\n"
            "- PARAGRAPHS & DEPTH: Write comprehensively. Break your analysis into multiple distinct, well-developed paragraphs within each section.\n"
            "- SUB-HEADERS: Frequently use Markdown sub-headers (###) or bold inline headers (**Topic:**) to organize complex data and metrics.\n"
            "- VALUATION FOCUS: Discuss the stock's 'fair value' and analyst 'price targets' or 'target price' across multiple sections (e.g., in Fundamental Analysis and Executive Summary), not just at the end.\n"
            "- FINAL VERDICT: In the Final Recommendation, you MUST explicitly state a clear action ('Buy', 'Sell', or 'Hold'), a specific 'Price Target', and a 'Time Horizon' (e.g., '12-month')."
        ),
    },
    TaskType.CS_DEBATE_ORCHESTRATION: {
        "description": (
            "Orchestrate a rigorous, multi-round debate about '{stock_symbol}'. "
            "YOU MUST FOLLOW THIS EXACT PROTOCOL FOR 3 ROUNDS:\n\n"
            "ROUND X START:\n"
            "1. Delegate to 'Sceptic': Instruct them to use 'read_current_context', find weaknesses in the analysts' claims, and use 'add_claims_to_context' to attack them. Tell Sceptic they MUST use the `refutes_id` field pointing to the claim they are attacking.\n"
            "2. Wait for Sceptic to finish.\n"
            "3. Delegate to 'Trust-Builder': Instruct them to use 'read_current_context' to find the Sceptic's new attacks. Tell Trust-Builder they MUST use 'add_claims_to_context' to defend the data, and they ABSOLUTELY MUST set the `refutes_id` field to the ID of the Sceptic's claim they are replying to.\n"
            "4. Wait for Trust-Builder to finish.\n"
            "ROUND X END.\n\n"
            "Repeat this for exactly 3 rounds. Do not stop early."
        ),
        "expected_output": (
            "A detailed log of the 3 rounds. Format:\n"
            "Round 1: Sceptic attacked [IDs], Trust defended [IDs].\n"
            "Round 2: ...\n"
            "Round 3: ..."
        ),
    },
}


def create_task(
    task_type: TaskType,
    agent: Optional[Agent] = None,
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
