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
    context_storage,
    read_current_context,
    add_fact_to_context,
    add_claim_to_context
)
from src.tools.qdrant_tools import (
    store_session_evidence, 
    query_session_evidence, 
    qdrant_service
)


class GroupChatV1StockAnalysisCrew:
    """Group Chat mode - 6 agents in hierarchical debate with Leader orchestrating."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.llm = LLM(
            model=self.config.advanced_model,
            api_key=self.config.api_key,
            temperature=self.config.temperature,
        )
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize all 6 agents for group chat mode."""
        # Original 3 specialist agents
        self.researcher = create_researcher_agent(self.llm)
        self.technical_analyst = create_technical_analyst_agent(self.llm)
        self.fundamental_analyst = create_fundamental_analyst_agent(self.llm)

        # Agents for debate
        self.sceptic = create_sceptic_agent_v1(self.llm)
        self.trust_agent = create_trust_agent_v1(self.llm)

        # Reporter/Strategist agent with access to context storage for final synthesis
        self.reporter = create_reporter_agent(self.llm)

        # Leader agent to orchestrate the process
        self.leader = create_leader_agent_v1(self.llm)

        def add_tools_to_agent(agent, tools_to_add):
            if agent.tools is None:
                agent.tools = []
            
            if isinstance(tools_to_add, list):
                agent.tools.extend(tools_to_add)
            else:
                agent.tools.append(tools_to_add)

        ctx_tools = [read_current_context, add_fact_to_context, add_claim_to_context]
        
        context_storage_agents = [
            self.researcher,
            self.technical_analyst,
            self.fundamental_analyst,
            self.sceptic,
            self.trust_agent,
            self.reporter,
        ]
        for agent in context_storage_agents:
            add_tools_to_agent(agent, ctx_tools)

        add_tools_to_agent(self.researcher, store_session_evidence)
        add_tools_to_agent(self.technical_analyst, store_session_evidence)
        add_tools_to_agent(self.fundamental_analyst, store_session_evidence)

        add_tools_to_agent(self.sceptic, query_session_evidence)
        add_tools_to_agent(self.trust_agent, query_session_evidence)
        add_tools_to_agent(self.reporter, query_session_evidence)

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

        sceptic_task = create_task(TaskType.CS_SCEPTIC, self.sceptic)
        trust_task = create_task(TaskType.CS_TRUST, self.trust_agent)

        reporting_task = create_task(TaskType.CS_REPORTING, self.reporter)

        # synthesis_task = create_task(TaskType.SYNTHESIS, self.leader)

        crew = Crew(
            agents=[
                self.researcher,
                self.technical_analyst,
                self.fundamental_analyst,
                self.sceptic,
                self.reporter,
                self.trust_agent,
            ],
            tasks=[
                research_task,
                technical_task,
                fundamental_task,
                sceptic_task,
                trust_task,
                # synthesis_task,
                reporting_task,
            ],
            manager_agent=self.leader,
            process=Process.hierarchical,
            verbose=True,
            cache=True,
        )

        try:
            result = crew.kickoff(inputs={"stock_symbol": stock_symbol.upper()})
            execution_time = time() - start_time

            return {
                "mode": "group_chat",
                "provider": self.config.provider.value,
                "execution_time": execution_time,
                "report": str(result),
                "context_data": context_storage.storage,
            }
        except Exception as e:
            execution_time = time() - start_time
            return {
                "mode": "group_chat",
                "provider": self.config.provider.value,
                "execution_time": execution_time,
                "report": f"Group Chat mode failed: {str(e)}. Recommend using Sequential mode.",
                "error": str(e),
            }
