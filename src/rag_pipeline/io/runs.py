import json 
import os 


def write_json(path: str, payload: dict) -> str:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w', encoding= 'utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path 