import json
import uuid
from typing import Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class FactEntry(BaseModel):
    content: str = Field(
        ...,
        description="The hard fact or numerical data",
        examples=["Revenue grew by 15% YoY."],
    )


class ClaimEntry(BaseModel):
    content: str = Field(
        ...,
        description="Your conclusion, interpretation, or counter-argument",
        examples=["High P/E ratio suggests overvaluation."]
    )
    refutes_id: Optional[str] = Field(None, description="ID of the entry you are challenging.")


class ContextStorage:
    """Central context storage from agents (Blackboard) - File Based."""

    def __init__(self, default_path: str = "current_context.json"):
        self.file_path = default_path

    def initialize_session(self, stock_symbol: str):
        """
        Sets a unique file path for the stock and ensures a fresh,
        empty JSON structure exists.
        """
        self.file_path = f".context/ctx_{stock_symbol.lower()}.json"
        initial_data: dict[str, list] = {"facts": [], "claims": []}
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

    def add_facts(self, agent_name: str, facts: list[FactEntry]) -> str:
        data = self._load_from_file()
        added_ids = []
        for f in facts:
            if isinstance(f, dict):
                content = f.get('content')
            else:
                content = getattr(f, 'content', None)

            fact_id = f"fact_{uuid.uuid4().hex[:8]}"
            data["facts"].append({
                "id": fact_id,
                "agent": agent_name,
                "content": content,
            })
            added_ids.append(fact_id)
        self._save_to_file(data)
        return f"Success: Added {len(added_ids)} facts: {', '.join(added_ids)}"

    def add_claims(self, agent_name: str, claims: list[ClaimEntry]) -> str:
        data = self._load_from_file()
        added_ids = []
        for c in claims:
            content = c.get('content') if isinstance(c, dict) else getattr(c, 'content', None)
            r_id = c.get('refutes_id') if isinstance(c, dict) else getattr(c, 'refutes_id', None)
            
            claim_id = f"claim_{uuid.uuid4().hex[:8]}"
            data["claims"].append({
                "id": claim_id,
                "agent": agent_name,
                "content": content,
                "refutes_id": r_id,
            })
            added_ids.append(claim_id)

        self._save_to_file(data)
        return f"Success: Added {len(added_ids)} claims."

    def get_context(self) -> str:
        """Returns the entire storage as a formatted JSON string."""
        data = self._load_from_file()
        return json.dumps(data, separators=(',', ':'))

    @property
    def storage(self):
        """Property to access the dictionary directly (e.g. for Streamlit)."""
        return self._load_from_file()


# --- TOOLS FOR CREWAI ---


class AddFactInput(BaseModel):
    agent_name: str = Field(..., description="Your role name (e.g., 'Senior Stock Market Researcher')")
    facts: list[FactEntry] = Field(..., description="List of facts to add at once")


class AddClaimInput(BaseModel):
    agent_name: str = Field(..., description="Your role name (e.g., 'Devil's Advocate - Sceptic')")
    claims: list[ClaimEntry] = Field(..., description="List of claims to add at once")


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
            "Saves a list of hard facts to the shared JSON file. "
            "Input MUST be a list of objects where each object has a 'content' key. "
            "Example: facts=[{'content': 'Fact 1'}, {'content': 'Fact 2'}]"
        )
        args_schema: Type[BaseModel] = AddFactInput

        def _run(self, agent_name: str, facts: list[FactEntry]) -> str:
            return storage_instance.add_facts(agent_name, facts)

    class AddClaimTool(BaseTool):
        name: str = "Add Claim to Context"
        description: str = (
            "Saves a list of interpretations or objections. "
            "Input MUST be a list of objects where each object has a 'content' key. "
            "Example: claims=[{'content': 'Claim 1'}, {'content': 'Claim 2'}]"
        )
        args_schema: Type[BaseModel] = AddClaimInput

        def _run(self, agent_name: str, claims: list[ClaimEntry]) -> str:
            return storage_instance.add_claims(agent_name, claims)

    return [ReadContextTool(), AddFactTool(), AddClaimTool()]
