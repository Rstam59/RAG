import os
import shutil
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from rag_pipeline.utils.hashing import get_file_hash

logger = logging.getLogger(__name__)

class ImageArtifactHandler:
    """
    Şəkilləri idarə edən ağıllı handler.
    Artıq məzmun dublikatlarını (Content Duplication) da tutur.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.deletion_dir = os.path.join(output_dir, "deleted_images")
        
        # YENİ: Sessiya ərzində görülən şəkillərin reyestri
        # Format: { "hash_string": { "filename": "...", "path": "..." } }
        self.seen_images: Dict[str, Dict[str, Any]] = {}
        
        self._prepare_dirs()

    def _prepare_dirs(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.deletion_dir):
            os.makedirs(self.deletion_dir)

    def save_image(self, image_bytes: bytes, proposed_filename: str) -> Dict[str, Any]:
        """
        Şəkli yadda saxlayır. 
        Əgər bu şəkil (hash-ə görə) əvvəl qarşımıza çıxıbsa, 
        yeni fayl yaratmır, köhnənin yolunu qaytarır.
        """
        # 1. Şəklin unikal imzasını (hash) alırıq
        current_hash = get_file_hash(image_bytes)

        # 2. DİQQƏT: Bu şəkli biz artıq görmüşükmü?
        if current_hash in self.seen_images:
            existing_info = self.seen_images[current_hash]
            logger.info(f"♻️ Content Duplicate found! Reuse: {existing_info['filename']} instead of {proposed_filename}")
            # Fiziki fayl yaratmırıq, köhnə şəklin məlumatını qaytarırıq
            return existing_info

        # 3. Əgər təzə şəkildirsə, davam edirik
        target_path = os.path.join(self.output_dir, proposed_filename)
        
        # (Köhnə fayl adı toqquşması yoxlanışı - ehtiyat üçün)
        if os.path.exists(target_path):
             with open(target_path, "rb") as f:
                existing_file_hash = get_file_hash(f.read())
             
             # Adı eynidir, amma içi başqadırsa -> Konflikt var, arxivləyirik
             if existing_file_hash != current_hash:
                 self._archive_existing_file(target_path, proposed_filename)

        # 4. Fiziki olaraq yazırıq
        with open(target_path, "wb") as f:
            f.write(image_bytes)
        
        logger.info(f"✅ New Image saved: {target_path}")

        # 5. Metadata hazırlayırıq
        metadata = {
            "filename": proposed_filename,
            "path": target_path,
            "hash": current_hash
        }

        # 6. Yaddaşa yazırıq ki, növbəti dəfə bunu görəndə təkrar yazmayaq
        self.seen_images[current_hash] = metadata
        
        return metadata

    def _archive_existing_file(self, file_path: str, filename: str):
        """Köhnə faylı 'deleted_images' qovluğuna daşıyır."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_part, ext_part = os.path.splitext(filename)
        archived_name = f"{name_part}_conflict_{timestamp}{ext_part}"
        backup_path = os.path.join(self.deletion_dir, archived_name)
        shutil.move(file_path, backup_path)