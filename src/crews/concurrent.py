from time import time

from crewai import LLM, Crew

from src.config import LLMConfig, LLMProvider
from src.crews.agents_definitions import (
    create_fundamental_analyst_agent,
    create_reporter_agent,
    create_researcher_agent,
    create_technical_analyst_agent,
)
from src.crews.tasks_definitions import TaskType, create_task
from src.tools.context_storage_tools import (
    add_claim_to_context,
    add_fact_to_context,
    context_storage,
    read_current_context,
)
from src.tools.qdrant_tools import qdrant_service, query_session_evidence, store_session_evidence


class ConcurrentStockAnalysisCrew:
    """
    Concurrent execution mode - 3 specialists run in parallel and share state through
    the Context Storage Blackboard. Three rounds:
      R1 (async, parallel): each specialist gathers data, writes facts/evidence.
      R2 (async, parallel): each specialist reads what the others wrote and adds
          cross-awareness claims (with refutes_id when correcting peers).
      R3 (sync): Reporter synthesizes the final markdown report from the Blackboard.
    """

    def __init__(self, config: LLMConfig):
        self.config = config

        llm_kwargs = {
            "model": self.config.advanced_model,
            "api_key": self.config.api_key,
            "temperature": self.config.temperature,
        }
        if self.config.provider == LLMProvider.LOCAL:
            llm_kwargs["base_url"] = self.config.api_base
        self.llm = LLM(**llm_kwargs)

        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize the 4 agents and attach Context Storage + RAG tools."""
        self.researcher = create_researcher_agent(self.llm)
        self.technical_analyst = create_technical_analyst_agent(self.llm)
        self.fundamental_analyst = create_fundamental_analyst_agent(self.llm)
        self.reporter = create_reporter_agent(self.llm)

        def add_tools_to_agent(agent, tools_to_add):
            if agent.tools is None:
                agent.tools = []
            if isinstance(tools_to_add, list):
                agent.tools.extend(tools_to_add)
            else:
                agent.tools.append(tools_to_add)

        ctx_tools = [read_current_context, add_fact_to_context, add_claim_to_context]
        for agent in (self.researcher, self.technical_analyst, self.fundamental_analyst, self.reporter):
            add_tools_to_agent(agent, ctx_tools)

        # Specialists store raw evidence; reporter queries it during synthesis.
        add_tools_to_agent(self.researcher, store_session_evidence)
        add_tools_to_agent(self.technical_analyst, store_session_evidence)
        add_tools_to_agent(self.fundamental_analyst, store_session_evidence)
        add_tools_to_agent(self.reporter, query_session_evidence)

    def run(self, stock_symbol: str) -> dict:
        """
        Run the 3-round concurrent crew and return the report with metadata.

        Args:
            stock_symbol: Stock ticker symbol

        Returns:
            Dictionary with mode, provider, execution_time, report and context_data
        """
        start_time = time()

        context_storage.initialize_session(stock_symbol)
        qdrant_service.initialize_session(stock_symbol)

        # Round 1 - parallel data gathering
        r1_research = create_task(TaskType.CS_RESEARCH, self.researcher, async_execution=True)
        r1_technical = create_task(TaskType.CS_TECHNICAL, self.technical_analyst, async_execution=True)
        r1_fundamental = create_task(TaskType.CS_FUNDAMENTAL, self.fundamental_analyst, async_execution=True)

        round_1 = [r1_research, r1_technical, r1_fundamental]

        # Round 2 - parallel cross-awareness refine, gated on the entire Round 1
        r2_research = create_task(
            TaskType.CONCURRENT_RESEARCH_REFINE,
            self.researcher,
            context=round_1,
            async_execution=True,
        )
        r2_technical = create_task(
            TaskType.CONCURRENT_TECHNICAL_REFINE,
            self.technical_analyst,
            context=round_1,
            async_execution=True,
        )
        r2_fundamental = create_task(
            TaskType.CONCURRENT_FUNDAMENTAL_REFINE,
            self.fundamental_analyst,
            context=round_1,
            async_execution=True,
        )

        round_2 = [r2_research, r2_technical, r2_fundamental]

        # Round 3 - synchronous synthesis, waits for both R1 and R2
        reporting_task = create_task(
            TaskType.CS_REPORTING,
            self.reporter,
            context=round_1 + round_2,
            async_execution=False,
        )

        crew = Crew(
            agents=[self.researcher, self.technical_analyst, self.fundamental_analyst, self.reporter],
            tasks=round_1 + round_2 + [reporting_task],
            cache=True,
            verbose=True,
        )

        try:
            result = crew.kickoff(inputs={"stock_symbol": stock_symbol.upper()})
            execution_time = time() - start_time

            return {
                "mode": "concurrent",
                "provider": self.config.provider.value,
                "execution_time": execution_time,
                "report": str(result),
                "context_data": context_storage.storage,
            }
        except Exception as e:
            execution_time = time() - start_time
            return {
                "mode": "concurrent",
                "provider": self.config.provider.value,
                "execution_time": execution_time,
                "report": f"Concurrent mode failed: {str(e)}. Recommend using Sequential mode.",
                "error": str(e),
            }
