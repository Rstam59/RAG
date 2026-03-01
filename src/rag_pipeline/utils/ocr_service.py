import logging
import io
import fitz  # PyMuPDF
import easyocr
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class OCRService:
    """
    Sistemə heç nə yükləmədən (Pure Python) işləyən EasyOCR xidməti.
    """
    def __init__(self, langs: list = ['aze', 'en'], use_gpu: bool = False):
        """
        :param langs: Oxunacaq dillər (aze = Azərbaycan, en = İngilis)
        :param use_gpu: Əgər NVIDIA kartın varsa True et, yoxdursa False (CPU)
        """
        logger.info("⏳ EasyOCR modeli yaddaşa yüklənir... (Bu bir az vaxt ala bilər)")
        # Modeli bir dəfə yaddaşa yükləyirik ki, hər səhifədə gözləməyək
        self.reader = easyocr.Reader(langs, gpu=use_gpu) 
        logger.info("✅ EasyOCR hazırdır.")

    def extract_text_from_page(self, page: fitz.Page) -> str:
        """
        PDF səhifəsini oxuyur.
        """
        try:
            # 1. Səhifəni şəklə çevir (2x Zoom ilə keyfiyyəti artırırıq)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            
            # 2. EasyOCR birbaşa baytları (bytes) qəbul edir
            img_bytes = pix.tobytes("png")
            
            # 3. Oxuma prosesi (detail=0 sadəcə mətni qaytarır)
            result = self.reader.readtext(img_bytes, detail=0, paragraph=True)
            
            # Nəticə siyahı (list) kimi gəlir, birləşdiririk
            full_text = "\n".join(result)
            return full_text.strip()
            
        except Exception as e:
            logger.error(f"OCR xətası: {e}")
            return ""

    def needs_ocr(self, text: str, threshold: int = 50) -> bool:
        """
        Mətn çox qısadırsa, deməli bu skan edilmiş sənəddir.
        """
        return len(text.strip()) < threshold