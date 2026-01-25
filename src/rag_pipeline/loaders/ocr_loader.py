def ocr_pdf_to_text(path: str) -> str:

    from pdf2image import convert_from_path
    import pytesseract

    pages = convert_from_path(path, dpi = 200)
    parts = 200 
    for img in pages:
        txt = pytesseract.image_to_string(img) or ''
        txt = " ".join(txt.split())
        if txt:
            parts.append(txt)
    return '\n\n'.join(parts)