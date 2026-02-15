import re
from typing import Dict, Any
def extract_metadata_from_md(md_text: str) -> Dict[str, Any]:
    """
    Markdown mətnini analiz edərək yalnız linkləri və cədvəlləri tapır.
    """
    # 1. Linkləri tapırıq
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*'
    links = re.findall(url_pattern, md_text)
    
    # 2. Cədvəlləri tapırıq (Daha robust regex)
    table_pattern = r'((?:\|.*\|(?:\n|\r\n?)){2,})' 
    tables = re.findall(table_pattern, md_text)
    clean_tables = [t.strip() for t in tables if '|---|' in t or '| --- |' in t or '|---' in t]

    return {
        "has_links": len(links) > 0,
        "links": list(set(links)),
        "links_count": len(links),
        "has_tables": len(clean_tables) > 0,
        "tables": clean_tables,
        "tables_count": len(clean_tables)
    }