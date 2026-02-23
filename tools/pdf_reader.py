"""
MRAgent — PDF Reader Tool
Extract text from PDF files with page references for study and research.

Created: 2026-02-23
"""

from pathlib import Path

from tools.base import Tool
from utils.logger import get_logger

logger = get_logger("tools.pdf_reader")

# Max characters per page to avoid context overflow
MAX_CHARS_PER_PAGE = 5000
MAX_TOTAL_CHARS = 50_000


class ReadPDFTool(Tool):
    """Read and extract text from PDF documents."""

    name = "read_pdf"
    description = (
        "Read a PDF file and extract its text content with page markers. "
        "Returns text organized by page: [Page 1] ... [Page 2] ... "
        "Good for reading research papers, documents, textbooks, and study materials. "
        "Supports page range selection."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the PDF file",
            },
            "start_page": {
                "type": "integer",
                "description": "First page to extract (1-indexed, default: 1)",
            },
            "end_page": {
                "type": "integer",
                "description": "Last page to extract (1-indexed, inclusive, default: all)",
            },
        },
        "required": ["path"],
    }

    def execute(self, path: str, start_page: int = None,
                end_page: int = None) -> str:
        filepath = Path(path).expanduser().resolve()

        if not filepath.exists():
            return f"❌ File not found: {filepath}"
        if not filepath.is_file():
            return f"❌ Not a file: {filepath}"
        if filepath.suffix.lower() != ".pdf":
            return f"❌ Not a PDF file: {filepath.name}"

        try:
            import PyPDF2
        except ImportError:
            return "❌ PyPDF2 not installed. Install with: pip install PyPDF2"

        self.logger.info(f"Reading PDF: {filepath}")

        try:
            reader = PyPDF2.PdfReader(str(filepath))
        except PyPDF2.errors.PdfReadError:
            return f"❌ Cannot read PDF (file may be corrupted): {filepath.name}"
        except Exception as e:
            return f"❌ Error opening PDF: {e}"

        total_pages = len(reader.pages)

        if total_pages == 0:
            return f"📄 {filepath.name} — PDF has no pages."

        # Check for encryption
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                return f"🔒 PDF is encrypted and cannot be read: {filepath.name}"

        # Resolve page range
        start = max(1, start_page or 1)
        end = min(total_pages, end_page or total_pages)

        if start > total_pages:
            return f"❌ Start page {start} exceeds total pages ({total_pages})"

        self.logger.info(f"Extracting pages {start}-{end} of {total_pages}")

        # Extract text
        lines = [f"📄 **{filepath.name}** ({total_pages} pages)\n"]
        total_chars = 0

        for page_num in range(start - 1, end):
            try:
                page = reader.pages[page_num]
                text = page.extract_text() or ""

                # Truncate per-page if huge
                if len(text) > MAX_CHARS_PER_PAGE:
                    text = text[:MAX_CHARS_PER_PAGE] + f"\n... (page truncated, {len(text)} total chars)"

                lines.append(f"\n--- [Page {page_num + 1}] ---\n")
                lines.append(text.strip() if text.strip() else "(no extractable text on this page)")

                total_chars += len(text)
                if total_chars > MAX_TOTAL_CHARS:
                    lines.append(f"\n... (output truncated at {MAX_TOTAL_CHARS:,} chars, "
                                 f"use start_page/end_page to read specific sections)")
                    break

            except Exception as e:
                lines.append(f"\n--- [Page {page_num + 1}] ---\n")
                lines.append(f"(error extracting page: {e})")

        # Summary footer
        pages_read = min(end - start + 1, len(lines))
        lines.append(f"\n---\n📊 Read {end - start + 1} of {total_pages} pages | {total_chars:,} characters extracted")

        return "\n".join(lines)
