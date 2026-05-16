import os
import uuid
import requests
from typing import Any, Dict, List
import litellm

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.config import get_qdrant_config, LLMConfig


class QdrantService:
    """Service to handle Session-scoped RAG operations in Qdrant using dynamic Embeddings."""

    def __init__(self):
        cfg = get_qdrant_config()
        self.client = QdrantClient(
            host=cfg.host,
            port=cfg.port,
            grpc_port=cfg.grpc_port,
            prefer_grpc=cfg.prefer_grpc
        )
        self.collection_name = None
        self.llm_config = LLMConfig(os.getenv("SELECTED_LLM_PROVIDER", "local"))

        # Determine dimensions and embedding function based on provider
        if self.llm_config.provider.value == "openai":
            self.embedding_dimensions = 1536
            self.embedding_model = "text-embedding-3-small"
        elif self.llm_config.provider.value == "gemini":
            self.embedding_dimensions = 768
            self.embedding_model = "gemini/text-embedding-004"
        else:
            # Fallback to local Ollama
            self.embedding_dimensions = 768
            self.embedding_model = "nomic-embed-text"

    def _get_embedding(self, text: str) -> List[float]:
        """Fetch embedding dynamically per provider."""
        if self.llm_config.provider.value == "local":
            url = f"{self.llm_config.api_base}/api/embeddings"
            payload = {"model": self.embedding_model, "prompt": text}
            try:
                response = requests.post(url, json=payload, timeout=60)
                response.raise_for_status()
                return response.json().get("embedding", [])
            except Exception as e:
                print(f"Error fetching local embedding: {e}")
                return []
        else:
            try:
                os.environ["OPENAI_API_KEY"] = self.llm_config.api_key
                response = litellm.embedding(
                    model=self.embedding_model,
                    input=text,
                    api_key=self.llm_config.api_key
                )
                return response.data[0]["embedding"]
            except Exception as e:
                print(f"Error fetching litellm embedding: {e}")
                return []

    def initialize_session(self, stock_symbol: str) -> str:
        """
        Creates a new temporary collection for the current analysis session.
        """
        session_id = uuid.uuid4().hex[:6]
        self.collection_name = f"session_{stock_symbol.lower()}_{session_id}"

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.embedding_dimensions, distance=Distance.COSINE),
        )
        return self.collection_name

    def add_evidence(self, text: str, metadata: Dict[str, Any]):
        """
        Chunks text, gets embeddings, and adds to the session collection.
        """
        if not self.collection_name:
            raise ValueError("Session not initialized. Call initialize_session first.")

        # Simple chunking logic
        chunks = [text[i:i+1000] for i in range(0, len(text), 800)]
        points = []

        for idx, chunk in enumerate(chunks):
            embedding = self._get_embedding(chunk)
            if embedding:
                points.append(
                    PointStruct(
                        id=uuid.uuid4().hex,
                        vector=embedding,
                        payload={**metadata, "document": chunk}
                    )
                )

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

    def search_evidence(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Searches for relevant evidence using Ollama embeddings.
        """
        if not self.collection_name:
            return []

        query_vector = self._get_embedding(query)
        if not query_vector:
            return []

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )

        return [
            {
                "content": res.payload.get("document", ""),
                "metadata": res.payload,
                "score": res.score
            }
            for res in results
        ]

    def get_all_evidence(self) -> list:
        """Retrieve all stored evidence from the current session's vector database."""
        if not self.collection_name:
            return []

        try:
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True,
                with_vectors=False
            )

            evidence_list = []
            for record in records:
                payload = record.payload or {}
                evidence_list.append({
                    "source": payload.get("source", "Unknown"),
                    "content": payload.get("document", "")
                })
            return evidence_list
        except Exception as e:
            print(f"Error fetching evidence: {e}")
            return []
