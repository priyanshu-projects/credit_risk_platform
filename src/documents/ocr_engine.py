import fitz  # PyMuPDF for rendering PDF to images
import easyocr
import numpy as np
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

class OCREngine:
    """
    Extracts text from scanned documents using EasyOCR.
    Used as a fallback when native PDF text extraction fails.
    """
    def __init__(self, languages=['en']):
        # Initialize EasyOCR reader once to save loading time
        # gpu=False assumes CPU constraints (8GB RAM constraint)
        logger.info("Initializing EasyOCR Model (This may take a moment on CPU...)")
        self.reader = easyocr.Reader(languages, gpu=False, verbose=False)

    def _pdf_page_to_image(self, page) -> np.ndarray:
        """
        Converts a PyMuPDF page object to a numpy array for EasyOCR.
        """
        # Render page to an image (pixmap)
        # Using scale 2.0 (matrix) improves OCR accuracy for small text
        zoom_matrix = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=zoom_matrix)
        
        # Convert pixmap to numpy array (RGB)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # EasyOCR expects RGB. If image has alpha channel (RGBA), slice it off.
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
            
        return img_array

    def extract_text(self, file_path: str | Path) -> Optional[str]:
        """
        Renders a PDF to images and extracts text using OCR.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        text_content = []
        try:
            with fitz.open(path) as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    
                    # 1. Convert PDF page to Image
                    img_array = self._pdf_page_to_image(page)
                    
                    # 2. Run EasyOCR
                    # detail=0 returns just the text strings without bounding box coordinates
                    # paragraph=True groups nearby lines into paragraphs
                    ocr_result = self.reader.readtext(img_array, detail=0, paragraph=True)
                    
                    if ocr_result:
                        page_text = "\n".join(ocr_result)
                        text_content.append(page_text)
            
            full_text = "\n\n--- Page Break ---\n\n".join(text_content).strip()
            return full_text if full_text else None
            
        except Exception as e:
            logger.error("Error during OCR on %s: %s", path, e)
            return None

if __name__ == "__main__":
    # Basic syntax check without downloading the heavy model yet
    logger.info("OCREngine defined successfully.")
