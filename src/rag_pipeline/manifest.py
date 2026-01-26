import json
import os
from datetime import datetime, timezone
from typing import Any, Dict


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_run_manifest(runs_dir: str, run_id: str, payload: Dict[str, Any]) -> str:
    os.makedirs(runs_dir, exist_ok=True)
    out = os.path.join(runs_dir, f"{run_id}.json")
    write_json(out, payload)
    return out
