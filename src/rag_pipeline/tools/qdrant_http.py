import httpx 


def qdrant_upsert_points(host: str, port: int, 
                         collection: str, points: list[dict], 
                         timeout_s: float = 60.0):
    url = 
