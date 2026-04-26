"""Streamlit login wrapper around streamlit-authenticator."""

import os
from pathlib import Path
from typing import Optional

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from dotenv import load_dotenv
from yaml.loader import SafeLoader

load_dotenv()

AUTH_CREDENTIALS_PATH = os.getenv("AUTH_CREDENTIALS_PATH", "./src/auth/credentials.yaml")


def _load_config() -> dict:
    path = Path(AUTH_CREDENTIALS_PATH)
    if not path.exists():
        st.error(
            f"Brakuje pliku z poświadczeniami: {path}. "
            f"Skopiuj `src/auth/credentials.example.yaml` do `{path}` i ustaw hasła."
        )
        st.stop()
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, Loader=SafeLoader)


def _build_authenticator() -> stauth.Authenticate:
    config = _load_config()
    return stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    )


def require_login() -> dict:
    """
    Render the login form and gate the rest of the app.

    Returns:
        Dict with at least {"username": ..., "name": ...} for the authenticated user.
        Calls st.stop() if the user is not authenticated.
    """
    if "_authenticator" not in st.session_state:
        st.session_state["_authenticator"] = _build_authenticator()
    authenticator: stauth.Authenticate = st.session_state["_authenticator"]

    try:
        authenticator.login(location="main")
    except Exception as e:
        st.error(f"Błąd logowania: {e}")
        st.stop()

    auth_status = st.session_state.get("authentication_status")
    if auth_status is False:
        st.error("Nieprawidłowa nazwa użytkownika lub hasło.")
        st.stop()
    if auth_status is None:
        st.warning("Zaloguj się, aby korzystać z platformy.")
        st.stop()

    return {
        "username": st.session_state.get("username"),
        "name": st.session_state.get("name"),
    }


def logout_button(location: str = "sidebar") -> None:
    """Render a logout button. Call after require_login()."""
    authenticator: Optional[stauth.Authenticate] = st.session_state.get("_authenticator")
    if authenticator is None:
        return
    try:
        authenticator.logout(location=location)
    except Exception:
        # Silenced - logout may render in different ways across stauth versions
        pass
