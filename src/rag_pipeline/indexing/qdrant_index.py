import uuid
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams


class QdrantIndex:
    def __init__(self, qdrant_url: str, collection: str) -> None:
        self.client = QdrantClient(url=qdrant_url)
        self.collection = collection

    def ensure_collection(self, dim: int) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert_batched(self, points: List[PointStruct], batch_size: int) -> None:
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=self.collection, points=batch)

    def make_points(
        self,
        *,
        doc_id: str,
        file_name: str,
        source_path: str,
        chunks: List[str],
        vectors: List[List[float]],
        payload_meta: Dict[str, Any],
    ) -> List[PointStruct]:
        """
        Deterministic point IDs: uuid5(doc_id:chunk_index).
        Eval is doc-level, but stable point IDs still matter for idempotency.
        """
        ns = uuid.UUID("12345678-1234-5678-1234-567812345678")

        points: List[PointStruct] = []
        for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
            pid = str(uuid.uuid5(ns, f"{doc_id}:{idx}"))
            payload = {
                "doc_id": doc_id,
                "file_name": file_name,
                "source_path": source_path,
                "chunk_index": idx,
                "text": chunk,
                **payload_meta,
            }
            points.append(PointStruct(id=pid, vector=vec, payload=payload))
        return points
