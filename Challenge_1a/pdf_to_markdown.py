import sys
import os
from pathlib import Path
import pdfplumber
import argparse
import re

def classify_heading(line):
    stripped = line.strip()
    upper = stripped.upper()
    # H1: HOPE To SEE You THERE!
    if upper == "HOPE TO SEE YOU THERE!":
        return f"# {stripped}"
    # H3: lines starting with these phrases
    elif any(upper.startswith(prefix) for prefix in [
        "PLEASE VISIT", "SO YOUR CHILD", "WWW."
    ]):
        return f"### {stripped}"
    # H2: all other ALL CAPS lines (except RSVP and the H1)
    elif re.match(r"^[A-Z0-9 ,\-()]+$", stripped) and len(stripped) > 2 and upper != "HOPE TO SEE YOU THERE!" and not upper.startswith("RSVP"):
        return f"## {stripped}"
    # RSVP line: keep as-is
    elif upper.startswith("RSVP"):
        return stripped
    else:
        return stripped
def pdf_to_markdown(pdf_path, md_path=None):
    """
    Convert a PDF file to a Markdown (.md) file using pdfplumber.
    Args:
        pdf_path (str): Path to the input PDF file.
        md_path (str, optional): Path to the output Markdown file. If not provided, uses same name as PDF.
    Returns:
        str: Path to the generated Markdown file.
    """
    if md_path is None:
        md_path = str(Path(pdf_path).with_suffix('.md'))
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                lines = text.splitlines()
                for i, line in enumerate(lines):
                    if not line.strip():
                        continue
                    formatted = classify_heading(line)
                    full_text += formatted + "\n\n"
            else:
                full_text += f"<!-- Page {page_num}: (No extractable text) -->\n\n"

    # Write to markdown file
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    return md_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a PDF to Markdown (.md) file.")
    parser.add_argument("pdf_path", help="Path to the input PDF file.")
    parser.add_argument("-o", "--output", help="Path to the output Markdown file.")
    args = parser.parse_args()

    if not os.path.isfile(args.pdf_path):
        print(f"Error: File '{args.pdf_path}' does not exist.")
        sys.exit(1)
    
    md_file = pdf_to_markdown(args.pdf_path, args.output)
    print(f"Markdown file created at: {md_file}")