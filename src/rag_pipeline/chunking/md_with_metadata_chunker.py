import re
import logging
from typing import List, Dict, Any
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def anchor_based_chunking(pages_list: List[Dict[str, Any]]) -> List[Document]:
    """
    Səhifələri birləşdirir, məntiqi başlıqlara görə bölür, 
    amma səhifə nömrələrini və metadatanı axınla (flow) izləyir.
    """
    if not pages_list:
        return []

    # 1. Splitter Konfiqurasiyası
    # DİQQƏT: Buraya 'Page_Anchor' əlavə ETMİRİK. 
    # Splitter yalnız mövzu başlıqlarına (H1-H3) görə böləcək.
    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3")
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False # Başlıqlar mətndə qalsın
    )

    # 2. "Stitching": Səhifələri markerlərlə birləşdiririk
    full_markdown = ""
    for page in pages_list:
        p_num = page['page']
        # Marker unikal və regex üçün rahat olmalıdır
        marker = f"\n###### PAGE_ID_{p_num}\n" 
        full_markdown += f"{marker}{page['content']}"

    # 3. Bölmə (Yalnız semantik)
    splits = splitter.split_text(full_markdown)

    final_chunks = []
    
    # Səhifə metadata lüğəti
    pages_dict = {p['page']: p['metadata'] for p in pages_list}
    
    # --- STATE MACHINE (YADDAŞ) ---
    # Başlanğıcda 1-ci səhifədə olduğumuzu fərz edirik
    current_active_pages = [1] 

    for split in splits:
        content = split.page_content
        
        # 4. Daxili Marker Analizi
        # Bu chunk-ın İÇİNDƏ yeni səhifəyə keçid varmı?
        # Regex bütün "PAGE_ID_X"ləri tapır.
        found_markers = re.findall(r'PAGE_ID_(\d+)', content)
        
        if found_markers:
            # Əgər marker varsa, deməli bu chunk həmin səhifələri əhatə edir.
            # Məsələn: Əvvəl 5-də idik, indi içəridə 6 çıxdı.
            # Deməli bu chunk həm 5-ə (başlanğıcı), həm 6-ya aiddir.
            new_pages = [int(m) for m in found_markers]
            
            # Cari konteksti yeniləyirik:
            # Köhnə səhifənin sonu + Yeni tapılan səhifələr
            # (Set istifadə edirik ki, təkrarlanmasın)
            combined_pages = list(set(current_active_pages + new_pages))
            
            # Növbəti chunk üçün "Last Seen" (sonuncu görülən) səhifəni yadda saxlayırıq.
            # Məntiq: Chunk-ın sonu hansı səhifə ilə bitibsə, növbəti chunk oradan davam edir.
            current_active_pages = [new_pages[-1]] 
            
            page_context = combined_pages
        else:
            # Əgər marker yoxdursa, deməli bu chunk tamamilə 
            # əvvəlki chunk-ın qaldığı səhifənin içindədir.
            page_context = current_active_pages

        # 5. Metadata Birləşdirilməsi
        # page_context içindəki bütün səhifələrin şəkillərini və linklərini yığırıq
        all_imgs = []
        all_links = []
        
        for p_num in page_context:
            if p_num in pages_dict:
                p_meta = pages_dict[p_num]
                all_imgs.extend([img['filename'] for img in p_meta.get("images", [])])
                all_links.extend(p_meta.get("links", []))

        combined_metadata = {
            "source": pages_list[0]['metadata']['source'],
            "page_numbers": sorted(list(set(page_context))), # [1] və ya [1, 2]
            "all_images": list(set(all_imgs)), # Unikal şəkillər
            "all_links": list(set(all_links)),
            "chunk_char_count": len(content),
            **split.metadata # Header_1, Header_2 bura gəlir
        }

        # 6. Təmizləmə (Cleanup)
        # Markeri mətndən silirik, çünki o yalnız bizim kod üçün lazım idi
        clean_content = re.sub(r'###### PAGE_ID_\d+\n?', '', content).strip()

        # Boş chunk-ları (məsələn, yalnız marker olan sətirləri) atırıq
        if len(clean_content) > 10:
            final_chunks.append(Document(
                page_content=clean_content,
                metadata=combined_metadata
            ))

    logger.info(f"Chunking bitdi. {len(final_chunks)} sənəd yaradıldı.")
    return final_chunks