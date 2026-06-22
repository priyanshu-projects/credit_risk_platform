import os
from pathlib import Path
from typing import Optional

from src.documents.pdf_parser import PDFParser
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DocumentRouter:
    """
    Orchestrates document text extraction.
    Attempts fast native PDF parsing first. If the document is scanned
    (yields little to no text), it falls back to EasyOCR.
    """
    def __init__(self, min_text_length: int = 50):
        self.pdf_parser = PDFParser()
        self.ocr_engine = None  # Lazy load to save memory
        self.min_text_length = min_text_length

    def _get_ocr_engine(self):
        """Lazy loads the OCR engine only when a scanned document is detected."""
        if self.ocr_engine is None:
            from src.documents.ocr_engine import OCREngine
            self.ocr_engine = OCREngine()
        return self.ocr_engine

    def process_document(self, file_path: str | Path) -> Optional[str]:
        """
        Main entry point for extracting text from any supported document.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        logger.info("Processing: %s", path.name)
        
        # 1. Try Native PDF Parsing
        text = self.pdf_parser.extract_text(path)
        
        # 2. Evaluate Native Text
        # If text is very short, it's likely just an embedded logo in a scanned PDF
        if text and len(text.strip()) >= self.min_text_length:
            logger.info(" -> Success: Native PDF text extracted.")
            return text
            
        # 3. Fallback to OCR for Scanned Documents
        logger.info(" -> Scanned document detected (or native extraction failed). Falling back to OCR...")
        ocr = self._get_ocr_engine()
        text = ocr.extract_text(path)
        
        if text and len(text.strip()) > 0:
            logger.info(" -> Success: OCR text extracted.")
            return text
            
        logger.warning(" -> Failed: Could not extract text via Native or OCR methods.")
        return None

if __name__ == "__main__":
    # import sys
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    # Basic syntax check
    router = DocumentRouter()
    logger.info("DocumentRouter initialized successfully.")

    files = [

        "sample_documents/loan_forms/loan_application_clean_native.pdf",

        "sample_documents/loan_forms/loan_application_messy_native.pdf",

        "sample_documents/loan_forms/loan_application_scanned.pdf",

        "sample_documents/loan_forms/sample_realistic_loan_application_native.pdf"

    ]

    for file in files:

        logger.info("\n" + "="*50)

        text = router.process_document(file)

        logger.info(text[:1000])
