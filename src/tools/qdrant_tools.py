import json

from crewai.tools import tool

from src.services.qdrant.service import QdrantService

qdrant_service = QdrantService()

@tool
def store_session_evidence(
    text: str,
    source: str,
    context_id: str
) -> str:
    """
    Saves a raw data snippet into the session-scoped vector database.
    Use this immediately after fetching data from APIs to preserve the 'proof'.

    Args:
        text (str): Raw text or data snippet to store.
        source (str): Source name (e.g., 'Yahoo Finance').
        context_id (str): ID from Context Storage this evidence supports.
    """
    try:
        metadata = {"source": source, "context_id": context_id}
        qdrant_service.add_evidence(text, metadata)
        return f"SUCCESS: Evidence linked to {context_id} stored."
    except Exception as e:
        return f"ERROR: {str(e)}"

@tool
def query_session_evidence(
    query: str,
    limit: int = 3
) -> str:
    """
    Searches the session's vector database for raw evidence snippets to verify claims.
    Returns a token-optimized JSON string of relevant findings.

    Args:
        query (str): Search query (e.g., 'revenue growth details').
        limit (int): Number of snippets to return.
    """
    try:
        results = qdrant_service.search_evidence(query, limit=limit)
        if not results:
            return "No evidence found."

        return json.dumps(results, separators=(',', ':'))
    except Exception as e:
        return f"ERROR: {str(e)}"
