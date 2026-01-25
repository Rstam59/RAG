import os 

def load_cache(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    
    with open(path, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}
    

def append_cache(path: str, key: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(key + '\n')