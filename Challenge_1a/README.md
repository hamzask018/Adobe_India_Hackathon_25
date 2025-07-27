# PDF to Markdown/Outline Extractor

This project processes PDF files and extracts their headings and structure into Markdown and a JSON outline, using font-size heuristics and vertical grouping.  
It is designed for use in Docker environments and does **not** require a machine learning classifier.

## Features

- Converts PDFs in `/app/input` to structured Markdown and outline JSON in `/app/output`
- Groups headings that are split across multiple lines (e.g., "HOPE TO SEE YOU THERE!")
- Robustly distinguishes headings from body text using font size and word count heuristics
- No dependencies on external classifiers or models

## Usage

### 1. Build the Docker Image

```sh
docker build -t pdf-outline-extractor .
```

### 2. Prepare Input and Output Folders

Make sure you have your PDF files in an `input` directory.  
Create an `output` directory to receive the results.

```sh
mkdir input output
cp yourfile.pdf input/
```

### 3. Run the Container

```sh
docker run --rm -v "$PWD/input":/app/input -v "$PWD/output":/app/output pdf-outline-extractor
```

- All PDFs in `/app/input` will be processed.
- For each PDF, a `.json` file is output to `/app/output` containing:
  - `title`: The main title extracted from the PDF
  - `outline`: A list of heading items with level, text, and page number

### 4. Output Example

```json
{
  "title": "Sample Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Sample Document Title",
      "page": 1
    },
    {
      "level": "H2",
      "text": "First Section",
      "page": 2
    }
    // ...
  ]
}
```

## File Structure

- `process_pdfs.py` — Main PDF processing script (uses only font-size heuristics)
- `requirements.txt` — Python dependencies (PyMuPDF, etc.)
- `README.md` — This file
- `Dockerfile` — For building the Docker image
- `/input` — Directory to place PDFs for processing (Docker volume)
- `/output` — Directory where results are written (Docker volume)

## Requirements

- Python 3.10+ (automatically provided in Docker)
- [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/) (`pip install pymupdf`)