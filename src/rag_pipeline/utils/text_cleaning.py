import re
import unicodedata
import ftfy


# 1. Görünməz "zibil" simvollar
_control_chars = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b\u200c\u200d\ufeff\ue000-\uf8ff]")

# 2. Səhifə nömrələri (Tək rəqəmləri silir)
_page_numbers = re.compile(r"^\s*\d+\s*$", re.MULTILINE)

# 3. Güllə başlıqları (•, ● -> -)
_bullets = re.compile(r"^\s*[•●▪]\s+", re.MULTILINE)

# 4. Standart boşluqlar (Space, Tab, NBSP)
_horizontal_ws = re.compile(r"[ \t\xA0\u1680\u180e\u2000-\u200a\u202f\u205f\u3000]+")

# 5. Paraqraf qoruyan boş sətirlər (3 və ya daha çox enteri tutur)
_multi_newlines = re.compile(r"\n{3,}")

# --- Spesifik Pattern-lər ---
_cid_pattern = re.compile(r"\(cid:[0-9]+\)")
_hyphenation = re.compile(r"(\w)-\n(\w)")

# 6. YENİ: Sınıq cümlələri birləşdirən "Smart Regex"
# Məntiq: Əgər sətir nöqtə, sual, nida ilə bitmirsə (?<![.:?!;])
# Və qarşıda yeni sətir (\n+) və kiçik hərf gəlirsə (?=[a-z]) -> Birləşdir!
_broken_paragraph = re.compile(r'(?<![.:?!;])\n+(?=[a-z])')


def clean_text(t: str) -> str:
    """
    Bütün fayl növləri (PDF, XLSX, DOCX) üçün universal və "Ağıllı" mətn təmizləyicisi.
    Paraqrafları qoruyur, sınıq cümlələri birləşdirir.
    """
    if not t:
        return ""

    # 1. Avtomatik Təmir (ftfy)
    t = ftfy.fix_text(t)

    # 2. Unicode Standartlaşdırma
    t = unicodedata.normalize("NFKC", t)

    # 3. Artefaktların Təmizlənməsi
    t = _control_chars.sub("", t)
    t = _cid_pattern.sub("", t)

    # 4. Səhifə nömrələrini və Bullet-ləri düzəldirik
    t = _page_numbers.sub("", t)
    t = _bullets.sub("- ", t)

    # 5. De-hyphenation (word-\nword -> wordword)
    # Bu, cümlə birləşməsindən ƏVVƏL olmalıdır!
    t = _hyphenation.sub(r"\1\2", t)

    # 6. YENİ: Sınıq cümlələri birləşdiririk (word\nword -> word word)
    # Bu addım sənin "due to the\nfact" problemini həll edir.
    t = _broken_paragraph.sub(" ", t)

    # 7. Səliqə-Sahman
    # Sətir daxili artıq boşluqları silirik
    t = _horizontal_ws.sub(" ", t)
    
    # 8. Paraqraf Strukturunu Qorumaq (ÇOX VACİB!)
    # Əvvəlki kodunda "splitlines" bütün \n\n-ləri pozurdu.
    # İndi biz sadəcə 3-dən çox olan boş sətirləri 2-yə endiririk.
    t = _multi_newlines.sub("\n\n", t)

    return t.strip()