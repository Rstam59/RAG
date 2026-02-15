import re 
from pypdf import PdfReader
from pypdf.errors import DependencyError, PdfReadError
from rag_pipeline.utils.text_cleaning import clean_text
import logging

logger = logging.getLogger("rag_pipeline.loaders.pdf_loader")

def read_pdf_text_best_effort(path: str) -> str: 
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as e:
                logger.warning(f"Could not decrypt PDF {path}: {e}")
                return ""
            
        parts = []
        for i, page in enumerate(reader.pages):
            try:
                raw = page.extract_text() or ""
                # ARTIQ BÜTÜN TƏMİZLİK İŞİNİ BU FUNKSİYA GÖRÜR
                txt = clean_text(raw)  
                if txt:
                    parts.append(txt)
            except Exception as e:
                logger.error(f"Error extracting page {i} from {path}: {e}")
                continue
        
        return '\n\n'.join(parts)

    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        return ""

