import re
import logging
from typing import List, Dict, Any, Union, Tuple
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class UniversalChunker:
    # --- PDF MARKERLƏRİ ÜÇÜN CONSTANTS ---
    MARKER_TEMPLATE = "\n\n###### PAGE_ID_{}\n\n"
    MARKER_REGEX = r'PAGE_ID_(\d+)'
    CLEANUP_REGEX = r'###### PAGE_ID_\d+\n?'
    MIN_CHUNK_LENGTH = 10

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 1. Semantik Bölücü (Markdown Başlıqları üçün)
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "Header_1"), ("##", "Header_2"), ("###", "Header_3")],
            strip_headers=False # Başlıqlar mətndə qalsın
        )

        # 2. Ölçü Bölücü (Çox böyük mətnlər üçün)
        self.rec_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )

    # ==========================================
    # 🔀 MAIN ROUTER (GİRİŞ QAPISI)
    # ==========================================
    def chunk(self, input_data: Union[str, List[Dict[str, Any]]]) -> List[Document]:
        """
        Giriş tipinə görə düzgün strategiyanı seçir.
        """
        # SENARİ 1: PDF LOADER-DƏN GƏLƏN SƏHİFƏLƏR (List[Dict])
        if isinstance(input_data, list):
            logger.info("Format: List[Dict] (PDF). Strategiya: Page-Aware Hierarchical Chunking")
            return self._process_hierarchical_pdf(input_data)

        # SENARİ 2: FULL TEXT STRING
        elif isinstance(input_data, str):
            logger.info("Format: String (Full Text). Strategiya: Markdown + Recursive Chunking")
            return self._process_full_text(input_data)
        
        else:
            raise ValueError("Input yalnız 'str' və ya 'List[Dict]' ola bilər.")

    # ==========================================
    # 📑 STRATEGY A: PAGE-AWARE HIERARCHICAL (PDF)
    # ==========================================
    def _process_hierarchical_pdf(self, pages_list: List[Dict[str, Any]]) -> List[Document]:
        """
        Səhifə nömrələrini, şəkilləri və cədvəlləri izləyərək bölür.
        """
        if not pages_list: return []

        # 1. Səhifələri markerlərlə "tikmək"
        full_text = "".join([
            f"{self.MARKER_TEMPLATE.format(p['page'])}{p['content']}" 
            for p in pages_list
        ])
        pages_dict = {p['page']: p['metadata'] for p in pages_list}

        # 2. Markdown Split
        md_chunks = self.md_splitter.split_text(full_text)
        
        final_documents = []
        last_known_page = 1

        for md_doc in md_chunks:
            # 3. Recursive Split (Lazım gələrsə)
            if len(md_doc.page_content) > self.chunk_size:
                sub_chunks_text = self.rec_splitter.split_text(md_doc.page_content)
            else:
                sub_chunks_text = [md_doc.page_content]

            # 4. Marker Analizi və Metadata
            for chunk_text in sub_chunks_text:
                # Səhifə izləmə
                found = re.findall(self.MARKER_REGEX, chunk_text)
                if found:
                    current_pages = sorted(list(set([int(m) for m in found])))
                    last_known_page = current_pages[-1]
                else:
                    current_pages = [last_known_page]

                # Təmizləmə
                clean_text = re.sub(self.CLEANUP_REGEX, '', chunk_text).strip()
                if len(clean_text) < self.MIN_CHUNK_LENGTH: continue

                # Metadata toplama
                aggregated_meta = self._collect_page_metadata(current_pages, pages_dict)
                
                # Document yaratma
                doc = Document(
                    page_content=clean_text,
                    metadata={
                        "page_numbers": current_pages,
                        "chunk_char_count": len(clean_text),
                        **aggregated_meta,
                        **md_doc.metadata # Header_1, Header_2
                    }
                )
                final_documents.append(doc)

        return final_documents

    # ==========================================
    # 📝 STRATEGY B: MARKDOWN + RECURSIVE (FULL TEXT)
    # ==========================================
    def _process_full_text(self, text: str) -> List[Document]:
        """
        Sadə mətnlər üçün: Əvvəl mövzuya (Markdown), sonra ölçüyə (Recursive) görə bölür.
        """
        if not text: return []

        # 1. Mərhələ: Semantik Bölmə (Markdown Başlıqları)
        # Əgər mətndə heç bir # yoxdursa, bütün mətni 1 parça kimi qaytarır.
        md_chunks = self.md_splitter.split_text(text)

        final_documents = []

        # 2. Mərhələ: Ölçü Yoxlanışı və Rekursiv Bölmə
        for md_doc in md_chunks:
            # md_doc.metadata içində artıq {'Header_1': 'Mövzu'} var
            
            if len(md_doc.page_content) > self.chunk_size:
                # Rekursiv splitter metadata-nı avtomatik miras (inherit) saxlayır!
                sub_chunks = self.rec_splitter.split_documents([md_doc])
                final_documents.extend(sub_chunks)
            else:
                # Ölçü kiçikdirsə, olduğu kimi saxla
                final_documents.append(md_doc)

        return final_documents

    # ==========================================
    # 🔧 HELPER METHODS
    # ==========================================
    def _collect_page_metadata(self, page_nums: List[int], pages_dict: Dict) -> Dict:
        """Səhifə metadatalarını (şəkil, link, cədvəl) birləşdirir."""
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