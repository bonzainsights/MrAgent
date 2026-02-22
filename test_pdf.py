from utils.logger import get_logger
import PyPDF2
from pathlib import Path

# Create a dummy PDF
from reportlab.pdfgen import canvas
c = canvas.Canvas("data/uploads/dummy.pdf")
c.drawString(100, 750, "Hello World from PDF")
c.save()

# Simulate the upload endpoint extract logic
file_path = Path("data/uploads/dummy.pdf")
text = ""
with open(file_path, "rb") as bf:
    pdf_reader = PyPDF2.PdfReader(bf)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"

print("Extracted PDF text:")
print(text.strip())
