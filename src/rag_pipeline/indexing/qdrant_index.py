import hashlib 
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()



def stable_point_id(file_hash:str, chunk_index:int) -> str:
    ns = uuid.UUID('12345678-1234-5678-1234-567812345678')
    return str(uuid.uuid5(ns, f"{file_hash}:{chunk_index}"))



class QdrantIndex:
    def __init__(self, host: str, port: int, collection: str) -> None:
        self.client = QdrantClient(host = host, port = port)
        self.collection = collection

    
    def ensure_collection(self, dim: int) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name = self.collection,
                vectors_config = VectorParams(size = dim, distance = Distance.COSINE)
            )


    def upsert_batched(self, points: list[PointStruct], batch_size: int) -> None:
        for i in range(0, len(points), batch_size):
            batch = points[i: i + batch_size]
            self.client.upsert(
                collection_name = self.collection, points = batch
            )

    def make_points(
            self,
            file_hash: str,
            file_name: str,
            source_path: str,
            chunks: list[str],
            vectors: list[list[float]]
) -> list[PointStruct]:
        
        points: list[PointStruct] = []
        for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
            points.append(
                PointStruct(
                    id = stable_point_id(file_hash, idx),
                    vector = vec,
                    payload = {
                        "source_path": source_path,
                        "file_name": file_name,
                        "file_hash": file_hash,
                        'chunk_index': idx,
                        'text': chunk
                    }
                )
            )
        return points
