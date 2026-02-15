import re
import logging
from typing import List, Dict, Any, Tuple
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class HierarchicalChunker:
    # --- CONSTANTS ---
    MARKER_TEMPLATE = "\n\n###### PAGE_ID_{}\n\n"
    MARKER_REGEX = r'PAGE_ID_(\d+)'
    CLEANUP_REGEX = r'###### PAGE_ID_\d+\n?'
    MIN_CHUNK_LENGTH = 10

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "Header_1"), ("##", "Header_2"), ("###", "Header_3")],
            strip_headers=False
        )
        self.rec_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )

    # ==========================================
    # 🚀 MAIN PROCESS (ORKESTRATOR)
    # ==========================================
    def process(self, pages_list: List[Dict[str, Any]]) -> List[Document]:
        """Prosesin əsas idarəetmə mərkəzi."""
        if not pages_list:
            return []

        # 1. Mətni birləşdir
        full_text = self._stitch_pages(pages_list)
        pages_dict = {p['page']: p['metadata'] for p in pages_list}
        
        # 2. Markdown parçalanması
        md_chunks = self.md_splitter.split_text(full_text)
        
        final_documents = []
        last_known_page = 1

        # 3. Hər bir Markdown bloku üzərində əməliyyat
        for md_doc in md_chunks:
            # Böyükdürsə xırdala, kiçikdirsə saxla
            sub_chunks_text = self._get_sub_chunks(md_doc.page_content)

            for chunk_text in sub_chunks_text:
                # Sənəd yarat və son səhifəni yenilə
                doc, last_known_page = self._create_single_document(
                    text=chunk_text,
                    last_known_page=last_known_page,
                    pages_dict=pages_dict,
                    parent_metadata=md_doc.metadata
                )
                
                if doc: # Əgər boş deyilsə əlavə et
                    final_documents.append(doc)

        logger.info(f"Chunking bitdi: {len(final_documents)} sənəd yaradıldı.")
        return final_documents

    # ==========================================
    # 🔧 HELPER METHODS (WORKERS)
    # ==========================================

    def _stitch_pages(self, pages: List[Dict[str, Any]]) -> str:
        """Səhifələri markerlərlə birləşdirir."""
        return "".join([
            f"{self.MARKER_TEMPLATE.format(p['page'])}{p['content']}" 
            for p in pages
        ])

    def _get_sub_chunks(self, text: str) -> List[str]:
        """Mətn böyükdürsə rekursiv bölür, yoxsa olduğu kimi qaytarır."""
        if len(text) > self.chunk_size:
            return self.rec_splitter.split_text(text)
        return [text]

    def _create_single_document(
        self, 
        text: str, 
        last_known_page: int, 
        pages_dict: Dict, 
        parent_metadata: Dict
    ) -> Tuple[Document | None, int]:
        """
        Tək bir chunk üçün: Səhifəni tapır, təmizləyir, metadata yığır və Document yaradır.
        Qaytarır: (Document obyekti, Yeni last_known_page)
        """
        # 1. Səhifə izləmə (State Tracking)
        current_pages, new_last_page = self._track_page_numbers(text, last_known_page)

        # 2. Təmizləmə
        clean_text = re.sub(self.CLEANUP_REGEX, '', text).strip()

        # 3. Validasiya (Çox qısadırsa atırıq)
        if len(clean_text) < self.MIN_CHUNK_LENGTH:
            return None, new_last_page

        # 4. Metadata toplama
        aggregated_meta = self._collect_metadata(current_pages, pages_dict)
        
        # 5. Document obyekti
        doc = Document(
            page_content=clean_text,
            metadata={
                "page_numbers": current_pages,
                "chunk_char_count": len(clean_text),
                **aggregated_meta,
                **parent_metadata
            }
        )
        return doc, new_last_page

    def _track_page_numbers(self, text: str, last_known: int) -> Tuple[List[int], int]:
        """Markerləri tapır və state yeniləyir."""
        found = re.findall(self.MARKER_REGEX, text)
        if found:
            pages = sorted(list(set([int(m) for m in found])))
            return pages, pages[-1]
        return [last_known], last_known

    def _collect_metadata(self, page_nums: List[int], pages_dict: Dict) -> Dict:
        """Səhifə metadatalarını (şəkil, link) birləşdirir."""
        all_imgs, all_links, all_tables = [], [], []
        for p in page_nums:
            if p in pages_dict:
                m = pages_dict[p]
                all_imgs.extend([img.get('filename') for img in m.get("images", [])])
                all_links.extend(m.get("links", []))
                all_tables.extend(m.get("tables", []))
        
        return {
            "all_images": list(set(all_imgs)),
            "all_links": list(set(all_links)),
            "has_tables": len(all_tables) > 0,
            "all_tables": list(set(all_tables))
        }