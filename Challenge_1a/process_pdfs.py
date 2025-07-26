import os
from pathlib import Path
from PyPDF2 import PdfReader

def process_pdfs():
    # Set input directory to sample_dataset/pdfs
    input_dir = Path("sample_dataset/pdfs")
    
    # Get all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    
    for pdf_file in pdf_files:
        print(f"\n--- Parsing {pdf_file.name} ---")
        try:
            reader = PdfReader(str(pdf_file))
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                print(f"\nPage {i+1}:\n{text if text else '[No text found]'}")
        except Exception as e:
            print(f"Error parsing {pdf_file.name}: {e}")

if __name__ == "__main__":
    print("Starting processing pdfs")
    process_pdfs()
    print("Completed processing")