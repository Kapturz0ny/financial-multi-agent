from crewai import LLM, Agent

from src.tools.alphavantage_tools import get_analyse_alphavantage_sentiment_tool
from src.tools.finnhub_sentiment_tool import get_analyse_finnhub_sentiment_tool
from src.tools.yahoo_analysis_tool import get_fetch_yahoo_analysis_tool
from src.tools.yahoo_fundamental_analysis_tool import get_analyse_fundamentals_tool
from src.tools.yahoo_news_tool import get_fetch_yahoo_news_tool
from src.tools.yahoo_technical_analysis_tool import get_analyse_technical_indicators_tool


def create_researcher_agent(llm: LLM, save_to_qdrant: bool = False) -> Agent:
    """Create Researcher agent for data gathering and sentiment analysis."""
    return Agent(
        role="Senior Stock Market Researcher",
        goal="Gather and analyze comprehensive data about {stock_symbol}",
        backstory="With a Ph.D. in Financial Economics and 15 years of experience in equity research, you're known for meticulous data collection and insightful analysis.",
        llm=llm,
        tools=[
            get_analyse_finnhub_sentiment_tool(save_to_qdrant=save_to_qdrant),
            get_analyse_alphavantage_sentiment_tool(save_to_qdrant=save_to_qdrant),
            get_fetch_yahoo_news_tool(save_to_qdrant=save_to_qdrant),
            get_fetch_yahoo_analysis_tool(save_to_qdrant=save_to_qdrant),
        ],
        verbose=True,
        memory=True,
        allow_code_execution=False,
        max_iter=8,
        max_execution_time=180,
    )


def create_technical_analyst_agent(llm: LLM, save_to_qdrant: bool = False) -> Agent:
    """Create Technical Analyst agent."""
    return Agent(
        role="Expert Technical Analyst",
        goal="Perform an in-depth technical analysis on {stock_symbol}",
        verbose=True,
        memory=True,
        backstory="As a Chartered Market Technician (CMT) with 15 years of experience, you have a keen eye for chart patterns and market trends.",
        tools=[
            get_analyse_technical_indicators_tool(save_to_qdrant=save_to_qdrant)
        ],
        llm=llm,
        allow_code_execution=False,
        max_iter=8,
        max_execution_time=180,
    )


def create_fundamental_analyst_agent(llm: LLM, save_to_qdrant: bool = False) -> Agent:
    """Create Fundamental Analyst agent."""
    return Agent(
        role="Senior Fundamental Analyst",
        goal="Conduct a comprehensive fundamental analysis of {stock_symbol}",
        verbose=True,
        memory=True,
        backstory="With a CFA charter and 15 years of experience in value investing, you dissect financial statements and identify key value drivers.",
        tools=[
            get_analyse_fundamentals_tool(save_to_qdrant=save_to_qdrant)
        ],
        llm=llm,
        allow_code_execution=False,
        max_iter=8,
        max_execution_time=180,
    )


def create_reporter_agent(llm: LLM) -> Agent:
    """Create Reporter/Investment Strategist agent."""
    return Agent(
        role="Chief Investment Strategist",
        goal="Synthesize all analyses to create a definitive investment report on {stock_symbol}",
        verbose=True,
        memory=True,
        backstory="As a seasoned investment strategist with 20 years of experience, you weave complex financial data into compelling investment narratives.",
        llm=llm,
        allow_code_execution=False,
        max_iter=8,
        max_execution_time=180,
    )


def create_sceptic_agent_v0(llm: LLM) -> Agent:
    """Create Sceptic (Devil's Advocate) agent for group chat."""
    return Agent(
        role="Devil's Advocate - Sceptic",
        goal="Challenge assumptions, identify risks, and point out potential weaknesses in the investment thesis for {stock_symbol}",
        backstory="You are a critical thinker who specializes in identifying hidden risks, flawed logic, and overlooked downsides. Your role is to ensure the team doesn't fall into confirmation bias.",
        llm=llm,
        tools=[],
        verbose=True,
        memory=True,
        allow_code_execution=False,
    )


def create_trust_agent_v0(llm: LLM) -> Agent:
    """Create Trust/Data Verification Specialist agent for group chat."""
    return Agent(
        role="Data Verification Specialist - Trust Builder",
        goal="Verify data accuracy, validate source credibility, and ensure all claims are well-substantiated for {stock_symbol}",
        backstory="You are meticulous about data integrity and source verification. Your expertise ensures that conclusions are grounded in reliable, well-documented evidence and not speculation.",
        llm=llm,
        tools=[],
        verbose=True,
        memory=True,
        allow_code_execution=False,
    )


def create_leader_agent_v0(llm: LLM) -> Agent:
    """Create Leader/Discussion Moderator agent for group chat."""
    return Agent(
        role="Discussion Moderator & Chief Synthesizer",
        goal="Moderate the team discussion, synthesize all perspectives into a cohesive final report for {stock_symbol}. ALWAYS end by asking for final advice from Trust Agent and Sceptic Agent before making final recommendation. Use all available agents to provide information for example Senior Stock Market Researcher, Expert Technical Analyst, Senior Fundamental Analyst.",
        backstory="As a senior investment strategist with 20 years of experience, you excel at extracting the strongest arguments from debate, integrating diverse perspectives, and forging consensus on a clear investment recommendation. Your approach: gather all perspectives → delegate to specialists → always consult Trust Agent for data verification → always consult Sceptic Agent for risks → synthesize into final recommendation.",
        llm=llm,
        tools=[],
        verbose=True,
        memory=True,
        allow_code_execution=False,
        allow_delegation=True,
    )


def create_sceptic_agent_v1(llm: LLM) -> Agent:
    return Agent(
        role="Sceptic",
        goal="Attack the investment thesis for {stock_symbol} by finding flaws in the Context Storage.",
        backstory=(
            "You are a ruthless devil's advocate. "
            "CRITICAL RULE: Whenever you add a claim using 'add_claims_to_context', you MUST link it to the specific fact or claim you are attacking. "
            "You do this by setting the `refutes_id` field in your JSON input. "
            "Example: [{'content': 'This P/E ratio is actually terrible compared to sector average.', 'refutes_id': 'claim_123'}]. "
            "If you do not include 'refutes_id', you fail your job."
        ),
        llm=llm,
        tools=[],
        verbose=True,
        memory=True,
        allow_code_execution=False,
    )


def create_trust_agent_v1(llm: LLM) -> Agent:
    return Agent(
        role="Trust-Builder",
        goal="Defend the data for {stock_symbol} against the Sceptic's attacks using hard evidence.",
        backstory=(
            "You are a meticulous data verifier. "
            "CRITICAL RULE: Your main job is to reply to the Sceptic. Whenever you use 'add_claims_to_context' to defend data, "
            "you ABSOLUTELY MUST set the `refutes_id` field to the exact ID of the Sceptic's claim you are responding to. "
            "Example: [{'content': 'Sceptic is wrong. Fact_456 proves cash flow is stable.', 'refutes_id': 'claim_999'}]. "
            "A defense without a 'refutes_id' is invalid."
        ),
        llm=llm,
        tools=[],
        verbose=True,
        memory=True,
        allow_code_execution=False,
    )


def create_leader_agent_v1(llm: LLM) -> Agent:
    """Create Leader/Discussion Moderator agent for group chat."""
    return Agent(
        role="Leader",
        goal=(
            "Orchestrate a dynamic, multi-round debate to stress-test information about {stock_symbol}. "
            "INSTRUCTIONS: "
            "1. Delegate work to the 'Sceptic' to attack the current findings. "
            "2. Then, delegate work to the 'Trust-Builder' to verify or refute the Sceptic's claims. "
            "3. YOU MUST CONDUCT AT LEAST 3 full debate rounds (Sceptic -> Trust-Builder -> Sceptic...). "
            "4. Do not finish the process until at least 3 rounds are complete and no new substantive claims are added."
        ),
        backstory=(
            "As a senior investment strategist with 20 years of experience, "
            "you excel at extracting the strongest arguments from debate, integrating diverse perspectives, "
            "and forging consensus on a clear investment recommendation. You strictly enforce debate rules."
        ),
        llm=llm,
        tools=[],
        verbose=True,
        memory=True,
        allow_code_execution=False,
    )
