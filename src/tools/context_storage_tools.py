from crewai.tools import tool

from src.services.context_storage.service import ClaimEntry, ContextStorage, FactEntry

# Global instance to be initialized in the Crew's run method
context_storage = ContextStorage()


@tool
def read_current_context() -> str:
    """
    Reads the shared JSON memory containing all facts and claims gathered so far from the file.
    Use this to review what other agents have found before adding your own claims or writing reports.
    """
    return context_storage.get_context()


@tool
def add_fact_to_context(agent_name: str, facts: list[FactEntry]) -> str:
    """
    Saves a list of hard facts to the shared JSON file.
    Input MUST be a list of objects where each object has a 'content' key.
    Example: facts=[{'content': 'Fact 1'}, {'content': 'Fact 2'}]

    Args:
        agent_name (str): Your exact role name (e.g., 'Senior Stock Market Researcher').
        facts (list[FactEntry]): List of facts to add at once.
    """
    return context_storage.add_facts(agent_name, facts)


@tool
def add_claim_to_context(agent_name: str, claims: list[ClaimEntry]) -> str:
    """
    Saves a list of interpretations or objections to the shared JSON file.
    Input MUST be a list of objects where each object has a 'content' key.
    Example: claims=[{'content': 'Claim 1'}, {'content': 'Claim 2', 'refutes_id': 'fact_123'}]

    Args:
        agent_name (str): Your exact role name (e.g., 'Devil's Advocate - Sceptic').
        claims (list[ClaimEntry]): List of claims to add at once.
    """
    return context_storage.add_claims(agent_name, claims)
