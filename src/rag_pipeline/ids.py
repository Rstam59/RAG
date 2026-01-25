import hashlib

def sha256_file(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors='ignore')).hexdigest()


def doc_id_from_file_hash(file_hash: str) -> str:
    return file_hash


def chunk_id(doc_id: str, chunker_id: str, chunk_text: str) -> str:
    normalized = " ".join(chunk_text.split())
    return sha256_text(f"{doc_id}|{chunker_id}|{normalized}")