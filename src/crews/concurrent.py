from time import time
from concurrent.futures import ThreadPoolExecutor

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

    # Hard caps to prevent runaway behavior in the parallel rounds.
    PER_TASK_TIMEOUT_S = 300  # ThreadPoolExecutor future timeout per specialist task
    CREW_MAX_RPM = 50         # CrewAI per-Crew RPM throttle (mitigates 429s)

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

    def _run_parallel_round(self, task_specs: list[tuple], inputs: dict):
        """Run a round in parallel by using one single-task crew per specialist.

        Each specialist gets its own Crew throttled by max_rpm. Hard timeout on
        each future guarantees we never wait forever if a tool/LLM call hangs.
        """

        def _run_single(agent, task_type):
            task = create_task(task_type, agent, async_execution=False)
            crew = Crew(
                agents=[agent],
                tasks=[task],
                cache=True,
                verbose=False,
                max_rpm=self.CREW_MAX_RPM,
            )
            return crew.kickoff(inputs=inputs)

        max_workers = len(task_specs)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_single, agent, task_type) for task_type, agent in task_specs]
            for future in futures:
                future.result(timeout=self.PER_TASK_TIMEOUT_S)

    def run(self, stock_symbol: str) -> dict:
        """
        Run the 3-round concurrent crew and return the report with metadata.

        Args:
            stock_symbol: Stock ticker symbol

        Returns:
            Dictionary with mode, provider, execution_time, report and context_data
        """
        start_time = time()
        inputs = {"stock_symbol": stock_symbol.upper()}

        context_storage.initialize_session(stock_symbol)
        qdrant_service.initialize_session(stock_symbol)

        # Round 1 - parallel data gathering (separate single-task crews)
        round_1_specs = [
            (TaskType.CS_RESEARCH, self.researcher),
            (TaskType.CS_TECHNICAL, self.technical_analyst),
            (TaskType.CS_FUNDAMENTAL, self.fundamental_analyst),
        ]

        # Round 2 - parallel cross-awareness refine (separate single-task crews)
        round_2_specs = [
            (TaskType.CONCURRENT_RESEARCH_REFINE, self.researcher),
            (TaskType.CONCURRENT_TECHNICAL_REFINE, self.technical_analyst),
            (TaskType.CONCURRENT_FUNDAMENTAL_REFINE, self.fundamental_analyst),
        ]

        # Round 3 - synchronous synthesis after both parallel rounds complete
        reporting_task = create_task(
            TaskType.CS_REPORTING,
            self.reporter,
            async_execution=False,
        )

        reporting_crew = Crew(
            agents=[self.reporter],
            tasks=[reporting_task],
            cache=True,
            verbose=False,
            max_rpm=self.CREW_MAX_RPM,
        )

        try:
            self._run_parallel_round(round_1_specs, inputs)
            self._run_parallel_round(round_2_specs, inputs)
            result = reporting_crew.kickoff(inputs=inputs)
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
