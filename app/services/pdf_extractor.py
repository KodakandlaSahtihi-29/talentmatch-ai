"""PDF and Document text extraction service using PyMuPDF (fitz)."""
import io
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> Tuple[str, int]:
    """
    Extracts text page-by-page from a PDF byte stream using PyMuPDF.
    
    Returns:
        Tuple[str, int]: (extracted text, total page count)
    Raises:
        ValueError: If PDF is empty, unreadable, or contains no extractable text.
    """
    if not pdf_bytes or len(pdf_bytes) == 0:
        raise ValueError("Provided PDF file is empty (0 bytes).")
    
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError("PyMuPDF (fitz) is not installed. Please run pip install pymupdf.")
    
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        logger.error(f"Failed to open PDF stream: {e}")
        raise ValueError(f"Corrupted or invalid PDF format: {str(e)}")
    
    pages_text = []
    total_pages = len(doc)
    
    if total_pages == 0:
        doc.close()
        raise ValueError("PDF document contains 0 pages.")
        
    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        if text.strip():
            pages_text.append(text.strip())
            
    doc.close()
    
    combined_text = "\n\n".join(pages_text).strip()
    
    if not combined_text:
        raise ValueError(
            "Could not extract any readable text from the PDF. "
            "The document may be a scanned image or protected."
        )
        
    return combined_text, total_pages

def extract_text_from_file(file_bytes: bytes, filename: str) -> Tuple[str, int]:
    """
    Dispatcher to extract text from PDF or plain text files.
    """
    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):
        return extract_text_from_pdf_bytes(file_bytes)
    elif lower_name.endswith((".txt", ".md", ".text")):
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1", errors="replace")
        if not text.strip():
            raise ValueError("Uploaded text file is empty.")
        return text.strip(), 1
    else:
        raise ValueError(f"Unsupported file format '{filename}'. Supported formats: PDF, TXT.")
