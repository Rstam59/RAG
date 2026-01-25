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



def read_pdf_text_strict(path: str, enable_ocr: bool) -> str:
    """
    Returns: (text, source)
        source: 'text' | 'ocr' | 'none'
    """

    text = read_pdf_text_best_effort(path)
    if text:
        return text, "text"
    
    if not enable_ocr:
        return "", 'none'
    

    try:
        from rag_pipeline.loaders.ocr_loader import ocr_pdf_to_text
        ocr_text = ocr_pdf_to_text(path)
        if ocr_text:
            return ocr_text, 'ocr'
        return "", 'none'
    except Exception:
        return '', 'none'