import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

class PDFParser:
    """
    Extracts raw text from native (digital) PDF documents.
    If a document is scanned (image-based), it will return None or an empty string, 
    which signals the pipeline to fall back to the OCR engine.
    """
    def __init__(self):
        pass

    def extract_text(self, file_path: str | Path) -> Optional[str]:
        """
        Reads a PDF and extracts all native text page by page.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        text_content = []
        try:
            # Open the PDF document
            with fitz.open(path) as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    # Extract text preserving basic layout blocks
                    text = page.get_text("text")
                    if text.strip():
                        text_content.append(text.strip())
            
            full_text = "\n\n--- Page Break ---\n\n".join(text_content).strip()
            
            # If no text was found (e.g., it's a scanned image), return None
            return full_text if full_text else None
            
        except Exception as e:
            logger.error("Error parsing PDF %s: %s", path, e)
            return None

if __name__ == "__main__":
    # Basic syntax check
    logger.info("PDFParser initialized successfully.")
