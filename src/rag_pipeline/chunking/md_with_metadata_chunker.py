import re
import logging
from typing import List, Dict, Any
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def anchor_based_chunking(pages_list: List[Dict[str, Any]]) -> List[Document]:
    if not pages_list:
        return []

    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
        ("######", "Page_Anchor") # Həm bizim markeri, həm də real H6-ları tutur
    ]

    # 1. Birləşdirmə
    full_markdown = ""
    for page in pages_list:
        p_num = page['page']
        # Markerimiz: ###### PAGE_ID_5
        marker = f"###### PAGE_ID_{p_num}"
        full_markdown += f"\n\n{marker}\n\n{page['content']}"

    # 2. Bölmə
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        strip_headers=False 
    )
    splits = splitter.split_text(full_markdown)

    final_chunks = []
    pages_dict = {p['page']: p['metadata'] for p in pages_list}
    
    # "Yaddaş": Əgər H6 bizim marker deyilsə, bu dəyişəndən istifadə edəcəyik
    current_page_num = 1 

    for split in splits:
        split_metadata = split.metadata
        anchor_val = split_metadata.get("Page_Anchor", "")
        
        # --- TƏHLÜKƏSİZLİK YOXLANIŞI (REGEX) ---
        # Yoxlayırıq: Bu başlıq bizim 'PAGE_ID_...' formatındadır?
        # Məsələn: 'PAGE_ID_5' -> Match!
        # Məsələn: 'Kiçik Başlıq' -> No Match!
        
        is_our_marker = False
        if anchor_val and "PAGE_ID_" in anchor_val:
            match = re.search(r'PAGE_ID_(\d+)', anchor_val)
            if match:
                current_page_num = int(match.group(1))
                is_our_marker = True
        
        # Əgər 'is_our_marker' False-dursa, deməli bu sadəcə mətndəki balaca bir başlıqdır.
        # Bu halda 'current_page_num' dəyişmir, əvvəlki səhifədə qaldığımızı fərz edirik.

        # --- METADATA ZƏNGİNLƏŞDİRMƏ ---
        page_info = pages_dict.get(current_page_num, {})
        
        combined_metadata = {
            "source": page_info.get("source"),
            "page_number": current_page_num,
            "all_images": [img['filename'] for img in page_info.get("images", [])],
            "all_links": page_info.get("links", []),
            "has_tables": page_info.get("has_tables", False),
            **split_metadata 
        }
        
        # --- TƏMİZLƏMƏ ---
        content = split.page_content
        
        if is_our_marker:
            # YALNIZ bizim markerdirsə silirik.
            # Əgər real H6-dırsa (məs: "###### Qeydlər"), ona toxunmuruq, mətndə qalır.
            # Regex: sətrin əvvəlində və ya ortasında ###### PAGE_ID_X varsa sil
            content = re.sub(r'######\s+PAGE_ID_\d+', '', content).strip()
            
            # Metadata-dan da o texniki "Page_Anchor" açarını silirik ki, bazanı zibilləməsin
            if "Page_Anchor" in combined_metadata:
                del combined_metadata["Page_Anchor"]

        if content:
            final_chunks.append(Document(
                page_content=content,
                metadata=combined_metadata
            ))

    logger.info(f"Chunking bitdi. {len(final_chunks)} sənəd yaradıldı.")
    return final_chunks