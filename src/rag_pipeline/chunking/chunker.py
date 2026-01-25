def chunk_text(text: str, chunk_chars: int, overlap: int, max_chunks: int = 0) -> list[str]:
    if not text:
        return []
    
    step = max(1, chunk_chars - overlap)
    chunks: list[str] = []

    for start in range(0, len(text), step):
        end = min(len(text), start + chunk_chars)
        c = text[start: end].strip()

        if c:
            chunks.append(c)

        if end >= len(text):
            break

        if max_chunks > 0 and len(chunks) >= max_chunks:
            break

    return chunks
    




