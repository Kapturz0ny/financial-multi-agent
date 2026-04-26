import uuid
from typing import Any, Dict, List

from qdrant_client import QdrantClient

from src.config import get_qdrant_config


class QdrantService:
    """Service to handle Session-scoped RAG operations in Qdrant."""

    def __init__(self):
        cfg = get_qdrant_config()
        self.client = QdrantClient(
            host=cfg.host,
            port=cfg.port,
            grpc_port=cfg.grpc_port,
            prefer_grpc=cfg.prefer_grpc
        )
        self.collection_name = None

    def initialize_session(self, stock_symbol: str) -> str:
        """
        Creates a new temporary collection for the current analysis session.
        The collection name includes a short UUID to ensure uniqueness.
        """
        session_id = uuid.uuid4().hex[:6]
        self.collection_name = f"session_{stock_symbol.lower()}_{session_id}"

        # We use FastEmbed (local) by default.
        # Qdrant client will automatically handle embedding if we use 'add' method.
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=self.client.get_fastembed_vector_params()
        )
        return self.collection_name

    def add_evidence(self, text: str, metadata: Dict[str, Any]):
        """
        Chunks text and adds it to the session collection with metadata.
        """
        if not self.collection_name:
            raise ValueError("Session not initialized. Call initialize_session first.")

        # Simple chunking logic
        chunks = [text[i:i+1000] for i in range(0, len(text), 800)]

        self.client.add(
            collection_name=self.collection_name,
            documents=chunks,
            metadata=[metadata] * len(chunks)
        )

    def search_evidence(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Searches for relevant evidence in the current session.
        """
        if not self.collection_name:
            return []

        results = self.client.query(
            collection_name=self.collection_name,
            query_text=query,
            limit=limit
        )

        return [
            {
                "content": res.document,
                "metadata": res.metadata,
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
