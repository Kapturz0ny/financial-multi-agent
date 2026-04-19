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

    # Tasks enhanced with Context Storage usage
    CS_RESEARCH = "cs_research"
    CS_TECHNICAL = "cs_technical"
    CS_FUNDAMENTAL = "cs_fundamental"
    CS_REPORTING = "cs_reporting"

    # Debate and synthesis tasks
    SCEPTIC = "sceptic"
    TRUST = "trust"
    SYNTHESIS = "synthesis"


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
            "Challenge assumptions made by other analysts. Be critical and thorough. "
            "CRITICAL: Gather ALL your findings first, then use the 'Add Claim to Context' tool "
            "ONCE or TWICE to save them in batches. Do not call the tool for every single news item."
            "Example: add_claim_to_context(agent_name='...', claims=[{'content': 'Claim A'}, {'content': 'Claim B'}])"
            "1. Use 'Read Current Context' to review all facts and claims submitted by the analysts.\n"
            "2. Critically analyze their assumptions. Look for flawed logic, ignored macroeconomic risks, or overly optimistic projections.\n"
            "3. Use 'Add Claim to Context' to record your objections. If you are directly challenging a specific claim made by another analyst, you MUST include its ID in the 'refutes_id' field.\n"
            "4. Use 'Add Claim to Context' to add alternative bearish scenarios or overlooked risks as new claims."
            "DO NOT summarize or list the claims in your final answer. "
            "The data is already safe in the Context Storage. Reporting it here will cause a system crash."
        ),
        "expected_output": (
            "SUCCESS: Critical analysis completed and saved to Context Storage."
        ),
    },
    TaskType.TRUST: {
        "description": (
            "Verify data credibility and validate claims about '{stock_symbol}'. "
            "Cross-check information across sentiment, technical, and fundamental analyses. "
            "Confirm source reliability and data accuracy. "
            "CRITICAL: Gather ALL your findings first, then use the 'Add Claim to Context' tool "
            "ONCE or TWICE to save them in batches. Do not call the tool for every single news item."
            "Example: add_claim_to_context(agent_name='...', claims=[{'content': 'Claim A'}, {'content': 'Claim B'}])"
            "1. Use 'Read Current Context' to review all entries in the database.\n"
            "2. Cross-check facts across the different analyses for consistency.\n"
            "3. Use 'Add Claim to Context' to validate strong, well-supported claims, or to flag suspicious, contradictory, or unverified data.\n"
            "4. If a fact or claim is demonstrably inaccurate or lacks reliability, use 'Add Claim to Context' with the 'refutes_id' pointing to that specific entry."
            "DO NOT summarize or list the claims in your final answer. "
            "The data is already safe in the Context Storage. Reporting it here will cause a system crash."
        ),
        "expected_output": (
            "SUCCESS: Data verification completed and saved to Context Storage."
        ),
    },
    TaskType.SYNTHESIS: {
        "description": (
            "You are the investment committee moderator. Your goal is to orchestrate the multi-agent analytical process for '{stock_symbol}'. "
            "1. Direct the Researcher, Technical Analyst, and Fundamental Analyst to populate the Context Storage with their specialized findings. "
            "2. Once the initial data is gathered, ensure the Sceptic challenges the assumptions and the Trust agent verifies the data integrity within the shared memory. "
            "3. Facilitate the discussion by ensuring all agents have contributed their facts and claims. "
            "4. Your task is complete when you have ensured that a full cycle of analysis, critique, and verification has been documented in the Context Storage by the respective specialists, providing a solid foundation for the final report."
        ),
        "expected_output": "A summary of the orchestration process and confirmation that all analytical perspectives (Research, Technical, Fundamental, Sceptic, Trust) have been documented in the Context Storage.",
    },
    TaskType.CS_RESEARCH: {
        "description": (
            "Gather and analyze qualitative data and public sentiment for '{stock_symbol}' by utilizing your specialized tools: "
            "1. Analyze broad market and social sentiment using **analyse_finnhub_sentiment** and **analyse_alphavantage_sentiment**. "
            "2. Fetch and process the most recent and impactful news articles using **fetch_yahoo_news**. "
            "3. Retrieve professional financial analyses, analyst ratings, and earnings consensus using **fetch_yahoo_analysis**. "
            "Focus on identifying the *drivers* of sentiment and the *impact* of news on the stock's outlook. "
            "CRITICAL: Gather ALL your findings first, then use the 'Add Fact to Context' tool "
            "ONCE or TWICE to save them in batches. Do not call the tool for every single news item."
            "Example: add_fact_to_context(agent_name='...', facts=[{'content': 'Fact A'}, {'content': 'Fact B'}])"
            "Use 'Add Fact to Context' to save hard data (e.g., specific news headlines, sentiment scores, exact analyst ratings). "
            "Use 'Add Claim to Context' to save your interpretations and syntheses of the gathered information."
            "DO NOT summarize or list the facts or claims in your final answer. "
            "The data is already safe in the Context Storage. Reporting it here will cause a system crash."
        ),
        "expected_output": (
            "SUCCESS: Sentiment and research data gathered and saved to Context Storage."
        ),
    },
    TaskType.CS_TECHNICAL: {
        "description": (
            "Perform an in-depth technical analysis of '{stock_symbol}' by fetching "
            "historical market data and calculating a wide array of technical indicators "
            "(using analyse_technical_indicators). Your primary goal is to **interpret these indicators** "
            "to identify trends, patterns, support/resistance levels, and potential trading signals, "
            "explaining their significance. "
            "CRITICAL: Gather ALL your findings first, then use the 'Add Fact to Context' tool "
            "ONCE or TWICE to save them in batches. Do not call the tool for every single new item."
            "Example: add_fact_to_context(agent_name='...', facts=[{'content': 'Fact A'}, {'content': 'Fact B'}])"
            "Use 'Add Fact to Context' to save specific numerical values (e.g., current price, exact support/resistance levels, specific indicator readings like 'RSI is 75'). "
            "Use 'Add Claim to Context' to save your interpretations of what these numbers mean."
            "DO NOT summarize or list the facts or claims in your final answer. "
            "The data is already safe in the Context Storage. Reporting it here will cause a system crash."
        ),
        "expected_output": (
            "SUCCESS: Technical analysis completed and saved to Context Storage."
        ),
    },
    TaskType.CS_FUNDAMENTAL: {
        "description": (
            "Conduct a comprehensive fundamental analysis of '{stock_symbol}' by fetching "
            "and analyzing its financial statements (income, balance sheet, cash flow), "
            "earnings reports, and company overview from Yahoo Finance (using analyse_fundamentals). "
            "Your focus is to **interpret this data** to assess its financial health, profitability, "
            "growth prospects, valuation, and overall intrinsic value, highlighting key strengths and weaknesses. "
            "CRITICAL: Gather ALL your findings first, then use the 'Add Fact to Context' tool "
            "ONCE or TWICE to save them in batches. Do not call the tool for every single new item."
            "Example: add_fact_to_context(agent_name='...', facts=[{'content': 'Fact A'}, {'content': 'Fact B'}])"
            "Use 'Add Fact to Context' to save hard financial metrics (e.g., specific P/E, P/S, Debt-to-Equity, ROE, profit margins, revenue figures). "
            "Use 'Add Claim to Context' to save your analytical conclusions."
            "DO NOT summarize or list the facts or claims in your final answer. "
            "The data is already safe in the Context Storage. Reporting it here will cause a system crash."
        ),
        "expected_output": (
            "SUCCESS: Fundamental analysis completed and saved to Context Storage."
        ),
    },
    TaskType.CS_REPORTING: {
        "description": (
            "Synthesize the sentiment analysis, technical analysis, and fundamental analysis for '{stock_symbol}' by reading the ENTIRE history of facts, claims, and refutations from the Context Storage. "
            "Your goal is to formulate a cohesive, insightful, and definitive investment report. **Do not simply concatenate previous entries.** "
            "Instead, integrate these diverse insights, critically evaluate them, highlight the most crucial findings and any convergences or divergences (especially those flagged by the Sceptic and Trust agents), discuss potential risks and rewards, and provide a clear, actionable investment thesis. "
            "Maintain a highly professional, academic, and sophisticated analytical tone. Use complex sentence structures and advanced financial terminology. "
            "Embed hard quantitative data throughout the text (exact numbers, percentages, dollar amounts, and dates) and explicitly cite sources based on the context."
        ),
        "expected_output": (
            "A comprehensive and well-structured textual investment report for '{stock_symbol}' formatted with clear Markdown headers (##). This report **must**:\n"
            "1.  Begin with a concise **Executive Summary** that states the overall investment thesis (Buy, Sell, Hold), a specific **Price Target**, a clear **Time Horizon**, and the key reasons supporting it.\n"
            "2.  **Synthesize and critically evaluate** key findings from the sentiment analysis. Highlight how public perception and news flow might impact the stock, going beyond just stating the sentiment.\n"
            "3.  **Synthesize and critically evaluate** key findings from the technical analysis. Explain how the technical outlook (indicators, patterns, support/resistance) aligns or contrasts with other analyses.\n"
            "4.  **Synthesize and critically evaluate** key findings from the fundamental analysis. Discuss the core financial health, valuation ratios, and growth prospects in the context of the overall thesis.\n"
            "5.  **Identify and discuss convergences or divergences** between the sentiment, technical, and fundamental analyses. Address how conflicts or refutations in the Context Storage were resolved.\n"
            "6.  Clearly outline the **primary catalysts** and a dedicated **Risk Assessment** section (incorporating the Sceptic's findings).\n"
            "7.  Conclude with a **well-reasoned investment outlook and a specific recommendation**, reiterating the main supporting points. The recommendation should be actionable and professional."
        ),
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
