import json
import uuid
from datetime import datetime
from typing import Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class ContextStorage:
    """Central context storage from agents (Blackboard) - File Based."""

    def __init__(self, default_path: str = "current_context.json"):
        self.file_path = default_path

    def initialize_session(self, stock_symbol: str):
        """
        Sets a unique file path for the stock and ensures a fresh,
        empty JSON structure exists.
        """
        self.file_path = f"./context/ctx_{stock_symbol.lower()}.json"
        initial_data = {"facts": [], "claims": []}
        self._save_to_file(initial_data)
        return f"Context storage initialized at {self.file_path}"

    def _load_from_file(self) -> dict:
        """Read the current state of the storage from the file."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"facts": [], "claims": []}

    def _save_to_file(self, data: dict):
        """Save the data dictionary to the JSON file."""
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add_fact(self, agent_name: str, content: str) -> str:
        data = self._load_from_file()
        fact_id = f"fact_{uuid.uuid4().hex[:8]}"
        fact = {
            "id": fact_id,
            "agent": agent_name,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        data["facts"].append(fact)
        self._save_to_file(data)
        return f"Success: Fact {fact_id} added to storage file."

    def add_claim(self, agent_name: str, content: str, refutes_id: Optional[str] = None) -> str:
        data = self._load_from_file()
        claim_id = f"claim_{uuid.uuid4().hex[:8]}"
        claim = {
            "id": claim_id,
            "agent": agent_name,
            "content": content,
            "refutes_id": refutes_id,
            "timestamp": datetime.now().isoformat(),
        }
        data["claims"].append(claim)
        self._save_to_file(data)
        return f"Success: Claim {claim_id} added to storage file."

    def get_context(self) -> str:
        """Returns the entire storage as a formatted JSON string."""
        data = self._load_from_file()
        return json.dumps(data, indent=2, ensure_ascii=False)

    @property
    def storage(self):
        """Property to access the dictionary directly (e.g. for Streamlit)."""
        return self._load_from_file()


# --- TOOLS FOR CREWAI ---

class AddFactInput(BaseModel):
    agent_name: str = Field(..., description="Your role name (e.g., 'Senior Stock Market Researcher')")
    content: str = Field(..., description="The hard fact or numerical data you found")

class AddClaimInput(BaseModel):
    agent_name: str = Field(..., description="Your role name (e.g., 'Devil's Advocate - Sceptic')")
    content: str = Field(..., description="Your conclusion, interpretation, or counter-argument")
    refutes_id: Optional[str] = Field(None, description="ID of a fact or claim you are refuting (optional)")


def create_context_storage_tools(storage_instance: ContextStorage):
    """Storage factory for particular memory instance."""

    class ReadContextTool(BaseTool):
        name: str = "Read Current Context"
        description: str = "Reads the shared JSON memory containing all facts and claims gathered so far from the file."

        def _run(self) -> str:
            return storage_instance.get_context()

    class AddFactTool(BaseTool):
        name: str = "Add Fact to Context"
        description: str = (
            "Saves a hard fact or numerical data to the shared JSON file. Use this instead of writing long reports."
        )
        args_schema: Type[BaseModel] = AddFactInput

        def _run(self, agent_name: str, content: str) -> str:
            return storage_instance.add_fact(agent_name, content)

    class AddClaimTool(BaseTool):
        name: str = "Add Claim to Context"
        description: str = "Saves your interpretation or objection to the shared JSON file. Use refutes_id if you disagree with an existing entry."
        args_schema: Type[BaseModel] = AddClaimInput

        def _run(self, agent_name: str, content: str, refutes_id: Optional[str] = None) -> str:
            return storage_instance.add_claim(agent_name, content, refutes_id)

    return [ReadContextTool(), AddFactTool(), AddClaimTool()]
