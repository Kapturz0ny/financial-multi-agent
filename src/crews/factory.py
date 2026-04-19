from enum import Enum
from typing import Optional

from src.config import load_config
from src.crews.group_chat import GroupChatStockAnalysisCrew
from src.crews.parallel import ParallelStockAnalysisCrew
from src.crews.sequential import SequentialStockAnalysisCrew


class CrewMode(Enum):
    """Enum for crew execution modes."""

    SEQUENTIAL = "sequential"
    GROUP_CHAT = "group_chat"
    PARALLEL = "parallel"


class StockAnalysisCrewFactory:
    """Factory for creating crew instances based on mode."""

    @staticmethod
    def create(mode: str, provider: Optional[str] = None):
        """
        Create a crew instance based on mode.

        Args:
            mode: CrewMode option
            provider: If None, uses LLM_PROVIDER from env.

        Returns:
            Instance of selected crew class

        Raises:
            ValueError: If mode is not recognized or provider is invalid
        """
        config = load_config(provider)

        if mode.lower() == CrewMode.SEQUENTIAL.value:
            return SequentialStockAnalysisCrew(config)
        elif mode.lower() == CrewMode.GROUP_CHAT.value:
            return GroupChatStockAnalysisCrew(config)
        elif mode.lower() == CrewMode.PARALLEL.value:
            return ParallelStockAnalysisCrew(config)
        else:
            raise ValueError(f"Unknown crew mode: {mode}. Use 'sequential' or 'group_chat'.")
