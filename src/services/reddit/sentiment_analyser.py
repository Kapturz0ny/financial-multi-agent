import json
import os
import requests
import litellm
from src.config import LLMConfig

class SentimentAnalyser:
    """
    A class to analyse sentiment using LLMs via network request or litellm.
    This replaces the local Transformers model to avoid AVX2 CPU lockups.
    """

    def __init__(self):
        # Default LLM provider context
        self.llm_config = LLMConfig(os.getenv("SELECTED_LLM_PROVIDER", "local"))
        self.labels = ["negative", "neutral", "positive"]

    def analyse(self, text: str) -> str:
        """
        Analyses the sentiment of the given text using the external LLM.
        """
        # Truncate text to avoid context limits
        if len(text) > 2000:
            text = text[:2000]

        if self.llm_config.provider.value == "local":
            return self._analyse_ollama(text)
        else:
            return self._analyse_litellm(text)

    def _analyse_litellm(self, text: str) -> str:
        """Use litellm to ask OpenAI/Gemini for sentiment."""
        prompt = (
            "Analyze the sentiment of the following text. "
            "Respond ONLY with one word: 'positive', 'neutral', or 'negative'. "
            f"Text: {text}"
        )
        try:
            os.environ["OPENAI_API_KEY"] = self.llm_config.api_key
            response = litellm.completion(
                model=self.llm_config.base_model,
                messages=[{"role": "user", "content": prompt}],
                api_key=self.llm_config.api_key,
                temperature=0.0
            )
            result = response.choices[0].message.content.strip().lower()
            if "positive" in result: return "positive"
            if "negative" in result: return "negative"
            return "neutral"
        except Exception as e:
            print(f"Error during sentiment analysis with litellm: {e}")
            return "neutral"

    def _analyse_ollama(self, text: str) -> str:
        url = f"{self.llm_config.api_base}/api/generate"
        prompt = (
            "Analyze the sentiment of the following text. "
            "Respond ONLY with one word: 'positive', 'neutral', or 'negative'. "
            f"Text: {text}"
        )
        
        payload = {
            "model": "qwen2.5:32b-instruct-q4_K_M",
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json().get("response", "").strip().lower()
            
            if "positive" in result: return "positive"
            if "negative" in result: return "negative"
            return "neutral"
        except Exception as e:
            print(f"Error during sentiment analysis with Ollama: {e}")
            return "neutral"
