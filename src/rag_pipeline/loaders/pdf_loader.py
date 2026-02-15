import datetime
import re
import shutil 
from pypdf import PdfReader
from pypdf.errors import DependencyError, PdfReadError
from rag_pipeline.utils.text_cleaning import clean_text
import logging
import os
import fitz  # PyMuPDF
import pymupdf4llm  # Yeni əlavə: Markdown çıxarışı üçün PyMuPDF wrapper
from rag_pipeline.utils.metadata_utils import extract_metadata_from_md
import os
import logging
from typing import List, Dict, Any
import hashlib

logger = logging.getLogger("rag_pipeline.loaders.pdf_loader")
def get_file_hash(file_bytes):
    """Şəklin məzmununa görə unikal MD5 hash yaradır."""
    return hashlib.md5(file_bytes).hexdigest()

def extract_pdf_with_images(path: str, output_img_dir: str = "data/extracted_images") -> list[dict[str, Any]]: 
    """
    PDF-i səhifə-səhifə oxuyur, şəkilləri çıxarır və mətndə onlara link verir.
    """
    deletion_dir = os.path.join(output_img_dir, "deleted_images")
    if not os.path.exists(output_img_dir):os.makedirs(output_img_dir)
    if not os.path.exists(deletion_dir): os.makedirs(deletion_dir)
    try:
        doc = fitz.open(path)
        if doc.is_encrypted:
            # Boş şifrə ilə yoxlayırıq (Əksər PDF-lərdə bu keçərli olur)
            if doc.authenticate(""):
                logger.info(f"PDF decrypted successfully with empty password: {path}")
            else:
                # Əgər real şifrə tələb olunursa, bura input və ya config-dən şifrə gələ bilər
                logger.error(f"PDF is password protected and cannot be opened: {path}")
                return []
            
        parts = []
        
        for i in range(len(doc)):
            try:
                md_text = pymupdf4llm.to_markdown(
                    doc, 
                    pages=[i], 
                    write_images=False, 
                    #write_images=True,  # Biz özümüz şəkilləri çıxarırıq, ona görə False qoyuruq
                    # image_path=output_img_dir,
                    # image_format="png"
                )
                # 2. Şəkilləri biz özümüz çıxarırıq (Metadata üçün tam nəzarət)
                img_list = doc[i].get_images(full=True)
                if not md_text.strip() and not img_list:
                    logger.warning(f"{path} Səhifə {i+1} boş qaldı (Şəkil və ya mətn), keçilir.")
                    continue
                img_metadatas = []
                img_markdown_links = ""
                base_name = os.path.basename(path).replace(".pdf", "")
                for img_idx, img in enumerate(img_list):
                    xref = img[0] # PyMuPDF-də hər bir şəkil "xref" nömrəsi ilə təmsil olunur yəni bu nömrə vasitəsilə şəkil məlumatlarına daxil ola bilərik
                    pix = doc.extract_image(xref) # Şəkil məlumatlarını çıxarırıq (bytes, format, width, height və s.)
                    new_img_bytes = pix["image"] # Şəkil məlumatlarından yalnız byte-ları alırıq
                    new_hash = get_file_hash(new_img_bytes) # Yeni şəkil üçün hash yaradırıq kim bu, mövcud şəkillərlə müqayisə üçün istifadə olunacaq
                    # Adlandırmanı biz edirik: sənəd_səhifə_sıra.ext
                    img_filename = f"{base_name}_p{i+1}_img{img_idx+1}.{pix['ext']}"
                    target_path = os.path.join(output_img_dir, img_filename)
                    # --- Robust Dublikat və Dəyişiklik Yoxlanışı ---

                    should_write = True
                    if os.path.exists(target_path):
                        with open(target_path, "rb") as f:
                            old_hash = get_file_hash(f.read())
                        
                        if old_hash == new_hash:
                            # Eyni şəkildirsə, heç nə etmə
                            should_write = False
                            logging.info(f"Duplicate image detected (same content) for {img_filename}. Skipping write.")
                        else:
                            # 1. Konflikt var: Köhnə faylı arxivə köçürürük
                            logger.warning(f"Conflict detected for {img_filename}. Archiving old version.")
                            
                            # 2. Vaxt möhürü yaradırıq (Məs: 20260215_165022)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            
                            # 3. Fayl adını və uzantısını ayırıb yeni ad veririk
                            name_part, ext_part = os.path.splitext(img_filename)
                            archived_name = f"{name_part}_deleted_{timestamp}{ext_part}"
                            
                            backup_path = os.path.join(deletion_dir, archived_name)
                            
                            # 4. Köhnəni yeni adla backup qovluğuna köçürürük
                            shutil.move(target_path, backup_path)
                    if should_write:
                        # Yeni şəkli əsas qovluğa yazırıq
                        with open(target_path, "wb") as f:
                            f.write(new_img_bytes)
                        logger.info(f"Image saved: {target_path}")
                    else:
                        logger.info(f"Image already exists and is identical: {target_path}. No action taken.")
                    # Metadata üçün şəkil məlumatlarını saxlayırıq
                    img_info = {
                            "filename": img_filename,
                            "path": target_path,
                            "hash": new_hash
                        }
                    img_metadatas.append(img_info)
                    # Markdown formatında link əlavə edirik
                    img_markdown_links += f"\n\n![Image]({target_path})\n"

                
                metadata_info = extract_metadata_from_md(md_text)
                content=clean_text(md_text).strip()
                full_content = content + img_markdown_links
                
                parts.append({
                    "page": i + 1,
                    "content": full_content,
                    "metadata": {
                        "source": path,
                        "img_dir": output_img_dir,
                        "has_images": len(img_metadatas) > 0,
                        "images": img_metadatas,
                        "image_count": len(img_metadatas),
                        **metadata_info  # Bütün link, table metadataları
                    }
                })
                logger.info(
                f"{path} Səhifə {i+1} emal edildi: "
                f"{len(content)} chars, "
                f"{metadata_info['links_count']} links, "
                f"{metadata_info['tables_count']} tables, "
                f"{len(img_metadatas)} images."  
            ) 
            except Exception as e:
                logger.warning(f"{path} Səhifə {i+1} emal edilə bilmədi: {e}")
                continue

        return parts

    except Exception as e:
        logger.error(f" {path}PDF xətası: {e}")
        return []
    finally:
        if doc:
            doc.close()

