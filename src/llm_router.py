"""Selects which LLM provider to use for the next analysis.

Rules:
- If the user explicitly chose `local`, honor it (no fallback needed, doesn't count toward quota).
- If the user is out of metered quota, fall back to `local` and signal a flag the UI can read.
- Otherwise, use the requested external provider.
"""

from typing import Optional

from src.auth.quota import remaining

LOCAL_PROVIDER = "local"


def select_provider(
    username: str,
    requested_provider: str,
    fallback_state: Optional[dict] = None,
) -> str:
    """
    Choose the provider to actually use.

    Args:
        username: authenticated username (for quota lookup).
        requested_provider: provider the user picked in the UI.
        fallback_state: optional mutable mapping (e.g. st.session_state) to set
            `fallback_active` / `fallback_reason` flags consumed by the UI.

    Returns:
        Effective provider string ("openai" | "gemini" | "local").
    """
    requested = (requested_provider or "").lower().strip()

    if requested == LOCAL_PROVIDER:
        if fallback_state is not None:
            fallback_state["fallback_active"] = False
            fallback_state["fallback_reason"] = None
        return LOCAL_PROVIDER

    if remaining(username) <= 0:
        if fallback_state is not None:
            fallback_state["fallback_active"] = True
            fallback_state["fallback_reason"] = (
                "Daily query limit for external models has been reached - "
                "switching to the Local Llama (Ollama) provider as fallback."
            )
        return LOCAL_PROVIDER

    if fallback_state is not None:
        fallback_state["fallback_active"] = False
        fallback_state["fallback_reason"] = None
    return requested
