import re
import logging
from typing import List, Dict, Any
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def hierarchical_chunking_strategy(
    pages_list: List[Dict[str, Any]], 
    chunk_size: int = 500, 
    chunk_overlap: int = 100
) -> List[Document]:
    """
    1. Səhifələri markerlə birləşdirir.
    2. Markdown başlıqlarına görə bölür (Mövzu).
    3. Əgər mövzu böyükdürsə, Recursive Splitter ilə ölçüyə görə bölür.
    4. Hər kiçik chunk-a düzgün səhifə nömrəsini təyin edir.
    """
    if not pages_list:
        return []

    # 1. Hazırlıq: Səhifələri markerlə "tikmək"
    full_text = ""
    for page in pages_list:
        p_num = page['page']
        # Marker: ###### PAGE_ID_5 (Önünə və sonuna newline qoyuruq ki, ayrılmasın)
        marker = f"\n\n###### PAGE_ID_{p_num}\n\n"
        full_text += f"{marker}{page['content']}"

    # 2. Mərhələ 1: Markdown Splitter (Mövzuya görə)
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header_1"), ("##", "Header_2"), ("###", "Header_3")],
        strip_headers=False
    )
    md_chunks = md_splitter.split_text(full_text)

    # 3. Mərhələ 2: Recursive Splitter (Ölçüyə görə)
    rec_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""], 
        keep_separator=True
    )

    final_documents = []
    pages_dict = {p['page']: p['metadata'] for p in pages_list}
    
    # Qlobal izləyici: Səhifə 1-dən başlayırıq
    last_known_page = 1

    # Hər bir Markdown parçasını (Topic) yoxlayırıq
    for md_doc in md_chunks:
        original_text = md_doc.page_content
        
        # Sual: Bu mövzu limitdən böyükdürmü?
        if len(original_text) > chunk_size:
            # Bəli -> Ölçüyə görə bölürük (Məs: 3 yerə)
            sub_chunks = rec_splitter.split_text(original_text)
        else:
            # Xeyr -> Olduğu kimi saxlayırıq
            sub_chunks = [original_text]

        # İndi alınan hər kiçik parçanı emal edirik
        for chunk_text in sub_chunks:
            # --- SƏHİFƏ ANALİZİ ---
            # Bu kiçik parçanın içində marker varmı?
            found_markers = re.findall(r'PAGE_ID_(\d+)', chunk_text)
            
            current_page_nums = []
            
            if found_markers:
                # Marker varsa, deməli yeni səhifəyə keçdik
                current_page_nums = sorted(list(set([int(m) for m in found_markers])))
                # İzləyicini yeniləyirik (sonuncu tapılan səhifə)
                last_known_page = current_page_nums[-1]
            else:
                # Marker yoxdursa, deməli əvvəlki səhifənin davamıdır
                current_page_nums = [last_known_page]

            # --- METADATA YIĞIMI ---
            all_imgs = []
            all_links = []
            all_tables = []
            
            for p in current_page_nums:
                if p in pages_dict:
                    meta = pages_dict[p]
                    all_imgs.extend([img['filename'] for img in meta.get("images", [])])
                    all_links.extend(meta.get("links", []))
                    all_tables.extend(meta.get("tables", []))

            # Markeri mətndən təmizləyirik
            clean_content = re.sub(r'###### PAGE_ID_\d+\n?', '', chunk_text).strip()
            
            # Boş chunk-ları atmırıq
            if len(clean_content) > 5:
                doc = Document(
                    page_content=clean_content,
                    metadata={
                        "source": pages_list[0]['metadata']['source'],
                        "page_numbers": current_page_nums, # [1] və ya [1, 2]
                        "all_images": list(set(all_imgs)),
                        "all_links": list(set(all_links)),
                        "has_tables": len(all_tables) > 0,
                        "chunk_char_count": len(clean_content),
                        **md_doc.metadata # Header_1, Header_2 bura gəlir
                    }
                )
                final_documents.append(doc)

    logger.info(f"Hierarchical Chunking bitdi: {len(md_chunks)} Markdown bloku -> {len(final_documents)} final chunk-a bölündü.")
    return final_documents