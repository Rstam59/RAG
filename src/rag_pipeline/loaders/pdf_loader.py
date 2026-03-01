import datetime
import re
import shutil 
from pypdf import PdfReader
from pypdf.errors import DependencyError, PdfReadError


import fitz  # PyMuPDF
import pymupdf4llm  # Yeni əlavə: Markdown çıxarışı üçün PyMuPDF wrapper
from rag_pipeline.utils.metadata_utils import extract_metadata_from_md
from rag_pipeline.utils.hashing import get_file_hash
from rag_pipeline.utils.text_cleaning import clean_text
from rag_pipeline.loaders.pdf_helpers import open_pdf_securely
from rag_pipeline.utils.image_handler import ImageArtifactHandler
from rag_pipeline.utils.ocr_service import OCRService
# Əgər 'pypdf' lazım dey
import os
import logging
from typing import List, Dict, Any


logger = logging.getLogger("rag_pipeline.loaders.pdf_loader")

def extract_pdf_with_images(path: str, output_img_dir: str = "data/extracted_images") -> List[Dict[str, Any]]: 
    """
    PDF-i səhifə-səhifə oxuyur, mətni Markdown-a çevirir, 
    şəkilləri diskə yazır və mətndə onlara istinad verir.
    """
    # 1. Handler-i başladırıq
    img_handler = ImageArtifactHandler(output_img_dir)
    ocr_service = OCRService()
    # 2. PDF-i təhlükəsiz açırıq
    doc = open_pdf_securely(path)
    if not doc:
        return []

    parts = []
    base_name = os.path.basename(path).replace(".pdf", "")

    try:
        # Səhifə dövrü: enumerate istifadə etmək daha təmizdir
        for i, page in enumerate(doc): 
            try:
                # 3. Mətni Markdown (Layout qorunmaqla) kimi çıxarırıq
                # write_images=False qoyuruq ki, şəkil emalını özümüz edək (dublikat kontrolu üçün)
                md_text = pymupdf4llm.to_markdown(doc, pages=[i], write_images=False)
                is_ocr_page = False
                # 4. Səhifədəki şəkilləri tapırıq
                if ocr_service.needs_ocr(md_text):
                    logger.info(f"Page {i+1} seems empty/scanned. Attempting OCR...")
                    ocr_text = ocr_service.extract_text_from_page(page)
                    
                    if ocr_text:
                        # OCR mətnini xüsusi başlıqla əlavə edirik
                        md_text = f"## [OCR Extracted Content]\n\n{ocr_text}"
                        is_ocr_page = True
                        logger.info(f"OCR successful for page {i+1}. Found {len(ocr_text)} chars.")
                    else:
                        logger.warning(f"Page {i+1} is empty even after OCR.")
                page_images = page.get_images(full=True)
                # Əgər həm mətn, həm də şəkil yoxdursa -> Boş səhifədir
                if not md_text.strip() and not page_images:
                    continue
                img_metadatas = []
                img_markdown_links = [] # String əvəzinə List istifadə etmək daha sürətlidir

                if not is_ocr_page:
                    for img_idx, img_info in enumerate(page_images):
                        xref = img_info[0] # Şəklin unikal ID-si
                    # DİQQƏT: extract_image Dictionary qaytarır, Pixmap yox! Adını 'base_image' qoyuruq
                    base_image = doc.extract_image(xref) 
                    if not base_image: # Şəkil xarabdırsa keçirik
                        continue

                    img_bytes = base_image["image"]
                    img_ext = base_image["ext"]
                    
                    img_filename = f"{base_name}_p{i+1}_img{img_idx+1}.{img_ext}"
                    
                    # Şəkli yaddaşa yazırıq (Handler dublikatları özü həll edir)
                    saved_meta = img_handler.save_image(img_bytes, img_filename)
                    
                    if saved_meta:
                        img_metadatas.append(saved_meta)
                        # Linki siyahıya yığırıq
                        img_markdown_links.append(f"![Image]({saved_meta['path']})")

                # 5. Mətn və Metadatanı birləşdiririk
                metadata_info = extract_metadata_from_md(md_text)
                clean_content = clean_text(md_text).strip()
                
                # Şəkilləri mətnin sonuna əlavə edirik (RAG üçün ən sadə yanaşma)
                # İki yeni sətir qoyuruq ki, mətndən ayrılsın
                full_content = clean_content + "\n\n" + "\n".join(img_markdown_links)

                parts.append({
                    "page": i + 1,
                    "content": full_content.strip(),
                    "metadata": {
                        "source": path,
                        "img_dir": output_img_dir,
                        "has_images": len(img_metadatas) > 0,
                        "images": img_metadatas,
                        "image_count": len(img_metadatas),
                        **metadata_info
                    }
                })
                
                logger.info(f"Processed page {i+1} of {path} | Chars: {len(clean_content)} | Imgs: {len(img_metadatas)}")

            except Exception as e:
                logger.warning(f"Error processing page {i+1} of {path}: {e}")
                continue
    except Exception as e:
        logger.error(f"Critical error reading PDF {path}: {e}")
        return []

    finally:
        # Sənədi mütləq bağlayırıq
        if doc:
            doc.close()

    return parts

#============ OLD VERSION (Şəkil çıxarışı olmadan) ============#
_ws = re.compile(r"\s+")
#old clean function
def _clean_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = _ws.sub(" ", t).strip()
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
            #Yeni: Təmizlənmiş mətni saxlayırıq, əgər boş deyilsə, onda əlavə edirik
            txt = clean_text(raw)
            if txt:
                parts.append(txt)
        
        return '\n\n'.join(parts)
    

    except (PdfReadError, DependencyError):
        print("PDF read error or dependency error")
        return ""
    except:
        return ""

