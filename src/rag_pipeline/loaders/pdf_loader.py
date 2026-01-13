import re 
from pypdf import PdfReader
from pypdf.errors import DependencyError, PdfReadError

_ws = re.compile(r"\s+")

def _clean_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = _ws.sub(t).strip()
    return t 


def read_pdf_text_best_effort(path: str) -> str: 
    """
    Best effort pdf extraction:

    - Handles encryiption attempt with empty password
    - Skips pages that crash
    - Returns "" for unreadable pages
    """

    try:
        reader = PdfReader(path)

        if getattr(reader, 'is_encrypted', False):
            try:
                reader.decrypt("")
            except Exception:
                return ""
            
        parts = []
        for page in reader.pages:
            try:
                raw = page.extract_text() or ""
            except Exception:
                continue
            txt = _clean_text(raw)

            if txt:
                parts.append(txt)
        
        return '\n\n'.join(parts)
    

    except (PdfReadError, DependencyError):
        return ""
    except:
        return ""

