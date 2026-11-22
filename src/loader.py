"""
Document loader: detect file type, extract text or render page images.

Returns a dict:
  input_mode : "text" | "vision"
  content    : str (text mode) | list[bytes] (vision mode, one PNG per page)

Auto-detection rule: if the PDF's total embedded-text character count exceeds
TEXT_CHAR_THRESHOLD the file is treated as a digital-text PDF; otherwise it is
treated as a scanned / image-only PDF and rendered to images.

The caller can override auto-detection with force_mode.
"""

from pathlib import Path

import pymupdf  # PyMuPDF >= 1.24 (import name changed from fitz)

# Minimum embedded-text characters to classify a PDF as "digital text"
TEXT_CHAR_THRESHOLD: int = 100

# Plain-image file extensions — always vision mode
_IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})


class LoadError(Exception):
    """Raised for corrupt, unsupported, empty, or unreadable input files."""


def load_document(path: str, force_mode: str | None = None) -> dict:
    """
    Load a document and return its content ready for the LLM.

    Args:
        path       : Absolute or relative path to a PDF or image file.
        force_mode : "text" or "vision" to override auto-detection.
                     Passing "text" on a plain image file raises LoadError.

    Returns:
        {
          "input_mode": "text" | "vision",
          "content"   : str              (text mode)
                      | list[bytes]      (vision mode, one PNG per page/image)
        }

    Raises:
        LoadError : file not found, empty, corrupt, or unsupported format.
    """
    p = Path(path)

    if not p.exists():
        raise LoadError(f"File not found: {path}")

    raw = p.read_bytes()
    if len(raw) == 0:
        raise LoadError(f"File is empty: {path}")

    ext = p.suffix.lower()

    if ext in _IMAGE_EXTS:
        return _load_image(raw, force_mode)
    if ext == ".pdf":
        return _load_pdf(raw, force_mode)

    raise LoadError(
        f"Unsupported file type '{ext}'. Supported extensions: "
        ".pdf, .jpg, .jpeg, .png, .bmp, .tiff, .webp"
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_image(raw: bytes, force_mode: str | None) -> dict:
    """Plain image file → always vision mode."""
    if force_mode == "text":
        raise LoadError("Cannot use text mode on a plain image file.")
    return {"input_mode": "vision", "content": [raw]}


def _load_pdf(raw: bytes, force_mode: str | None) -> dict:
    """PDF file → text or vision mode depending on embedded text."""
    try:
        doc = pymupdf.open(stream=raw, filetype="pdf")
    except Exception as exc:
        raise LoadError(f"Cannot open PDF (corrupt or invalid): {exc}") from exc

    if len(doc) == 0:
        raise LoadError("PDF has zero pages.")

    # Count total embedded text characters across all pages
    total_chars: int = sum(len(page.get_text()) for page in doc)

    mode = force_mode or ("text" if total_chars > TEXT_CHAR_THRESHOLD else "vision")

    if mode == "text":
        return _pdf_text(doc)
    return _pdf_vision(doc)


def _pdf_text(doc) -> dict:
    """Extract embedded text from every page and join with blank lines."""
    pages = [page.get_text() for page in doc]
    content = "\n\n".join(pages).strip()
    if not content:
        raise LoadError(
            "PDF has no readable embedded text. "
            "Re-run with --mode vision if this is a scanned document."
        )
    return {"input_mode": "text", "content": content}


def _pdf_vision(doc) -> dict:
    """Render every page to a PNG at 2× zoom and return the bytes."""
    zoom = pymupdf.Matrix(2.0, 2.0)  # 2× → ~144 dpi for better OCR quality
    images = [page.get_pixmap(matrix=zoom).tobytes("png") for page in doc]
    return {"input_mode": "vision", "content": images}

