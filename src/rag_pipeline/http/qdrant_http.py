import random
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx


class QdrantHttp:
    """
    Robust HTTP client for Qdrant operations we need for tools/eval:
      - search
      - scroll
    """
    def __init__(self, base_url: str, timeout_s: float = 20.0, max_retries: int = 3) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(timeout_s, connect=5.0)
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=self.timeout)

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, json_body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                r = self._client.request(method, url, json=json_body)
                r.raise_for_status()
                return r.json()
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as e:
                last_exc = e
                if attempt == self.max_retries:
                    break
                backoff = min(2.0 ** (attempt - 1), 8.0) + random.random() * 0.25
                time.sleep(backoff)

        raise RuntimeError(f"Qdrant HTTP request failed after {self.max_retries} retries: {last_exc}")

    def search(
        self,
        *,
        collection: str,
        vector: List[float],
        limit: int,
        with_payload: bool = True,
        filter_payload: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        body: Dict[str, Any] = {
            "vector": vector,
            "limit": limit,
            "with_payload": with_payload,
        }
        if filter_payload:
            body["filter"] = filter_payload

        data = self._request("POST", f"/collections/{collection}/points/search", json_body=body)
        return data.get("result", [])

    def scroll(
        self,
        *,
        collection: str,
        limit: int = 128,
        offset: Optional[Dict[str, Any]] = None,
        filter_payload: Optional[Dict[str, Any]] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        body: Dict[str, Any] = {
            "limit": limit,
            "with_payload": with_payload,
            "with_vectors": with_vectors,
        }
        if offset is not None:
            body["offset"] = offset
        if filter_payload:
            body["filter"] = filter_payload

        data = self._request("POST", f"/collections/{collection}/points/scroll", json_body=body)
        result = data.get("result", {}) or {}
        points = result.get("points", []) or []
        next_offset = result.get("next_page_offset")
        return points, next_offset
