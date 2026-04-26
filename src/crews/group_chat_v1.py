from time import time

from crewai import LLM, Crew
from crewai.process import Process

from src.config import LLMConfig
from src.crews.agents_definitions import (
    create_fundamental_analyst_agent,
    create_leader_agent_v1,
    create_reporter_agent,
    create_researcher_agent,
    create_sceptic_agent_v1,
    create_technical_analyst_agent,
    create_trust_agent_v1,
)
from src.crews.tasks_definitions import TaskType, create_task
from src.tools.context_storage_tools import (
    add_claims_to_context,
    add_facts_to_context,
    context_storage,
    read_current_context,
)
from src.tools.qdrant_tools import qdrant_service, query_session_evidence


class GroupChatV1StockAnalysisCrew:
    """Group Chat mode - 6 agents in hierarchical debate with Leader orchestrating."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_llm = LLM(
            model=self.config.base_model,
            api_key=self.config.api_key,
            temperature=self.config.temperature,
        )
        self.advanced_llm = LLM(
            model=self.config.advanced_model,
            api_key=self.config.api_key,
            temperature=self.config.temperature,
        )
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize all agents for group chat mode."""
        # Original 3 specialist agents
        self.researcher = create_researcher_agent(self.base_llm)
        self.technical_analyst = create_technical_analyst_agent(self.base_llm)
        self.fundamental_analyst = create_fundamental_analyst_agent(self.base_llm)

        # Agents for debate
        self.sceptic = create_sceptic_agent_v1(self.base_llm)
        self.trust_agent = create_trust_agent_v1(self.base_llm)

        # Reporter/Strategist agent with access to context storage for final synthesis
        self.reporter = create_reporter_agent(self.advanced_llm)

        # Leader agent to orchestrate the process
        self.leader = create_leader_agent_v1(self.base_llm)

        def add_tools_to_agent(agent, tools_to_add):
            if agent.tools is None:
                agent.tools = []

            if isinstance(tools_to_add, list):
                agent.tools.extend(tools_to_add)
            else:
                agent.tools.append(tools_to_add)


        add_tools_to_agent(self.researcher, [add_facts_to_context, add_claims_to_context])
        add_tools_to_agent(self.technical_analyst, [add_facts_to_context, add_claims_to_context])
        add_tools_to_agent(self.fundamental_analyst, [add_facts_to_context, add_claims_to_context])

        add_tools_to_agent(self.sceptic, [query_session_evidence, read_current_context, add_claims_to_context])
        add_tools_to_agent(self.trust_agent, [query_session_evidence, read_current_context, add_claims_to_context])
        add_tools_to_agent(self.reporter, [query_session_evidence, read_current_context, add_claims_to_context])

    def run(self, stock_symbol: str) -> dict:
        """
        Run group chat mode with hierarchical process - leader orchestrates 6 specialist agents.

        Args:
            stock_symbol: Stock ticker symbol

        Returns:
            Dictionary with mode, provider, execution_time, and report
        """
        start_time = time()

        context_storage.initialize_session(stock_symbol)
        qdrant_service.initialize_session(stock_symbol)

        research_task = create_task(TaskType.CS_RESEARCH, self.researcher)
        technical_task = create_task(TaskType.CS_TECHNICAL, self.technical_analyst)
        fundamental_task = create_task(TaskType.CS_FUNDAMENTAL, self.fundamental_analyst)

        debate_task = create_task(TaskType.CS_DEBATE_ORCHESTRATION)

        reporting_task = create_task(TaskType.CS_REPORTING, self.reporter)

        analysis_crew = Crew(
            agents=[
                self.researcher,
                self.technical_analyst,
                self.fundamental_analyst,
            ],
            tasks=[
                research_task,
                technical_task,
                fundamental_task,
            ],
            process=Process.sequential,
            verbose=False,
            cache=True,
        )

        debate_crew = Crew(
            agents=[
                self.sceptic,
                self.trust_agent,
            ],
            tasks=[debate_task],
            manager_agent=self.leader,
            process=Process.hierarchical,
            verbose=False,
        )


        try:
            # Perform analysis
            analysis_crew.kickoff(inputs={"stock_symbol": stock_symbol.upper()})

            # Debate
            debate_crew.kickoff(inputs={"stock_symbol": stock_symbol.upper()})

            # Finish with report
            report = Crew(
                agents=[self.reporter],
                tasks=[reporting_task],
                process=Process.sequential,
                verbose=False,
                cache=False,
            ).kickoff(inputs={"stock_symbol": stock_symbol.upper()})
            execution_time = time() - start_time

            return {
                "mode": "group_chat_v1",
                "provider": self.config.provider.value,
                "execution_time": execution_time,
                "report": str(report),
                "context_data": context_storage.storage,
            }
        except Exception as e:
            execution_time = time() - start_time
            return {
                "mode": "group_chat_v1",
                "provider": self.config.provider.value,
                "execution_time": execution_time,
                "report": f"Group Chat mode failed: {str(e)}. Recommend using other mode.",
                "error": str(e),
            }
