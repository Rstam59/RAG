import re
import unicodedata
import ftfy


# Görünməz "zibil" simvollar
_control_chars = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b\u200c\u200d\ufeff\ue000-\uf8ff]")

# Səhifə nömrələri (Adətən PDF və Word üçün)
_page_numbers = re.compile(r"^\s*\d+\s*$", re.MULTILINE)

# Güllə başlıqları (•, ● -> -)
_bullets = re.compile(r"^\s*[•●▪]\s+", re.MULTILINE)

# Standart boşluqlar
_horizontal_ws = re.compile(r"[ \t\xA0\u1680\u180e\u2000-\u200a\u202f\u205f\u3000]+")

# Paraqraf qoruyan boş sətirlər
_multi_newlines = re.compile(r"\n{3,}")

# --- Spesifik Pattern-lər (Əsasən PDF üçün, amma digərlərinə ziyanı yoxdur) ---
_cid_pattern = re.compile(r"\(cid:[0-9]+\)")
_hyphenation = re.compile(r"(\w)-\n(\w)")


def clean_text(t: str) -> str:
    """
    Bütün fayl növləri (PDF, XLSX, DOCX) üçün universal mətn təmizləyicisi.
    """
    if not t:
        return ""

    # 1. Avtomatik Təmir (ftfy) - Kodlaşdırma xətaları (Exceldə də olur!)
    t = ftfy.fix_text(t)

    # 2. Unicode Standartlaşdırma
    t = unicodedata.normalize("NFKC", t)

    # 3. Artefaktların Təmizlənməsi
    t = _control_chars.sub("", t)
    t = _cid_pattern.sub("", t)    # Excel-də olmasa da, ziyanı yoxdur

    # 4. Səhifə nömrələrini və Bullet-ləri düzəldirik
    t = _page_numbers.sub("", t)
    t = _bullets.sub("- ", t)

    # 5. De-hyphenation (Excel-də "Wrap Text" varsa lazım ola bilər)
    t = _hyphenation.sub(r"\1\2", t)

    # 6. Səliqə-Sahman
    t = _horizontal_ws.sub(" ", t)
    
    # Sətir kənarlarını təmizləyirik
    t = "\n".join(line.strip() for line in t.splitlines())

    # 7. Paraqraf strukturunu qoruyuruq
    t = _multi_newlines.sub("\n\n", t)

    return t.strip()