"""
loader.py — Document loader for regex_experiment.
Supports digital PDFs (text mode) and scanned PDFs / images (OCR mode).
If Tesseract is not installed, ocr-mode documents are flagged as ocr_unavailable.
"""

import os
import time

try:
    import pymupdf  # PyMuPDF >= 1.24
except ImportError:
    pymupdf = None

try:
    import pytesseract
    from PIL import Image
    import io
    _PYTESSERACT_AVAILABLE = True
except ImportError:
    _PYTESSERACT_AVAILABLE = False


MIN_TEXT_CHARS = 100  # same threshold as original pipeline


def _tesseract_available() -> bool:
    """Check whether the Tesseract binary is reachable."""
    if not _PYTESSERACT_AVAILABLE:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def load_document(path: str) -> dict:
    """
    Load a document and return:
      {
        "content": str | None,          # extracted text (or None for ocr_unavailable)
        "input_mode": "text" | "ocr" | "ocr_unavailable",
        "ocr_available": bool,
        "pages": int,
        "text_chars": int,
        "seconds": float,
      }
    """
    t0 = time.time()
    ext = os.path.splitext(path)[1].lower()

    if pymupdf is None:
        raise RuntimeError("pymupdf is not installed. Run: pip install pymupdf")

    # ------------------------------------------------------------------ images
    if ext in (".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".gif"):
        ocr_ok = _tesseract_available()
        if not ocr_ok:
            return {
                "content": None,
                "input_mode": "ocr_unavailable",
                "ocr_available": False,
                "pages": 1,
                "text_chars": 0,
                "seconds": time.time() - t0,
            }
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        return {
            "content": text,
            "input_mode": "ocr",
            "ocr_available": True,
            "pages": 1,
            "text_chars": len(text),
            "seconds": time.time() - t0,
        }

    # ------------------------------------------------------------------ PDFs
    if ext == ".pdf":
        doc = pymupdf.open(path)
        pages = doc.page_count

        # Try embedded text extraction
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text("text"))
        full_text = "\n".join(text_parts).strip()

        if len(full_text) >= MIN_TEXT_CHARS:
            doc.close()
            return {
                "content": full_text,
                "input_mode": "text",
                "ocr_available": True,  # doesn't matter, text is available
                "pages": pages,
                "text_chars": len(full_text),
                "seconds": time.time() - t0,
            }

        # Scanned PDF — need OCR
        ocr_ok = _tesseract_available()
        if not ocr_ok:
            doc.close()
            return {
                "content": None,
                "input_mode": "ocr_unavailable",
                "ocr_available": False,
                "pages": pages,
                "text_chars": 0,
                "seconds": time.time() - t0,
            }

        # Render each page to image and OCR
        ocr_texts = []
        for page in doc:
            mat = pymupdf.Matrix(2, 2)  # 2x zoom → ~144 DPI
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            ocr_texts.append(pytesseract.image_to_string(img))
        doc.close()
        combined = "\n".join(ocr_texts).strip()
        return {
            "content": combined,
            "input_mode": "ocr",
            "ocr_available": True,
            "pages": pages,
            "text_chars": len(combined),
            "seconds": time.time() - t0,
        }

    raise ValueError(f"Unsupported file type: {ext}")

