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


class ParallelStockAnalysisCrew:
    """Parallel execution mode - specialists work concurrently, reporter synthesizes at the end."""

    def __init__(self, config: LLMConfig):
        self.config = config
        llm_kwargs = {
            "model": self.config.base_model,
            "api_key": self.config.api_key,
            "temperature": 0.2,
        }
        if self.config.provider == LLMProvider.LOCAL:
            llm_kwargs["base_url"] = self.config.api_base
        self.llm = LLM(**llm_kwargs)
        self._initialize_agents_and_tasks()

    def _initialize_agents_and_tasks(self):
        """Initialize all agents and tasks for parallel mode."""
        researcher = create_researcher_agent(self.llm)
        technical_analyst = create_technical_analyst_agent(self.llm)
        fundamental_analyst = create_fundamental_analyst_agent(self.llm)
        reporter = create_reporter_agent(self.llm)

        # Specialists work parallel async_execution=True
        research_task = create_task(TaskType.RESEARCH, researcher, async_execution=True)
        technical_analysis_task = create_task(TaskType.TECHNICAL_ANALYSIS, technical_analyst, async_execution=True)
        fundamental_analysis_task = create_task(TaskType.FUNDAMENTAL_ANALYSIS, fundamental_analyst, async_execution=True)

        # Reporter works synchronously after specialists finish, async_execution=False
        reporting_task = create_task(
            TaskType.REPORTING,
            reporter,
            context=[research_task, technical_analysis_task, fundamental_analysis_task],
            async_execution=False
        )

        self.crew = Crew(
            agents=[researcher, technical_analyst, fundamental_analyst, reporter],
            tasks=[research_task, technical_analysis_task, fundamental_analysis_task, reporting_task],
            cache=True,
            verbose=True,
        )

    def run(self, stock_symbol: str) -> dict:
        """
        Run parallel crew and return report with metadata.

        Args:
            stock_symbol: Stock ticker symbol

        Returns:
            Dictionary with mode, provider, execution_time, and report
        """
        start_time = time()
        result = self.crew.kickoff(inputs={"stock_symbol": stock_symbol.upper()})
        execution_time = time() - start_time

        return {
            "mode": "parallel",
            "provider": self.config.provider.value,
            "execution_time": execution_time,
            "report": str(result),
        }
