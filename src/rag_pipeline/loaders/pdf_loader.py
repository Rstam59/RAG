import re
from typing import List

from pypdf import PdfReader
from pypdf.errors import DependencyError, PdfReadError

_ws = re.compile(r"\s+")


def clean_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = _ws.sub(" ", t).strip()
    return t


def read_pdf_text_best_effort(path: str) -> str:
    """
    Best-effort PDF text extraction:
    - decrypt with empty password if encrypted
    - skip pages that crash extraction
    - return "" if unreadable/no text
    """
    try:
        reader = PdfReader(path)

        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:
                return ""

        parts: List[str] = []
        for page in reader.pages:
            try:
                raw = page.extract_text() or ""
            except Exception:
                continue
            txt = clean_text(raw)
            if txt:
                parts.append(txt)

        return "\n\n".join(parts)

    except (DependencyError, PdfReadError):
        return ""
    except Exception:
        return ""
