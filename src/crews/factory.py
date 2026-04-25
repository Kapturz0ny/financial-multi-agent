from enum import Enum
from typing import Optional

from src.config import load_config
from src.crews.concurrent import ConcurrentStockAnalysisCrew
from src.crews.group_chat_v0 import GroupChatV0StockAnalysisCrew
from src.crews.group_chat_v1 import GroupChatV1StockAnalysisCrew
from src.crews.parallel import ParallelStockAnalysisCrew
from src.crews.sequential import SequentialStockAnalysisCrew


class CrewMode(Enum):
    """Enum for crew execution modes."""

    SEQUENTIAL = "sequential"
    GROUP_CHAT_V0 = "group_chat_v0"
    GROUP_CHAT_V1 = "group_chat_v1"
    PARALLEL = "parallel"
    CONCURRENT = "concurrent"


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
        elif mode.lower() == CrewMode.GROUP_CHAT_V0.value:
            return GroupChatV0StockAnalysisCrew(config)
        elif mode.lower() == CrewMode.GROUP_CHAT_V1.value:
            return GroupChatV1StockAnalysisCrew(config)
        elif mode.lower() == CrewMode.PARALLEL.value:
            return ParallelStockAnalysisCrew(config)
        elif mode.lower() == CrewMode.CONCURRENT.value:
            return ConcurrentStockAnalysisCrew(config)
        else:
            raise ValueError(f"Unknown crew mode: {mode}.")
