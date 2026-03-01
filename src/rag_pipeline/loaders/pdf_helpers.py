import fitz
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def open_pdf_securely(path: str) -> Optional[fitz.Document]:
    """PDF-i açır və şifrəli olub-olmadığını yoxlayır."""
    try:
        doc = fitz.open(path)
        if doc.is_encrypted:
            if doc.authenticate(""):
                logger.info(f"PDF decrypted with empty password: {path}")
            else:
                logger.error(f"PDF password protected: {path}")
                return None
        return doc
    except Exception as e:
        logger.error(f"Failed to open PDF {path}: {e}")
        return None