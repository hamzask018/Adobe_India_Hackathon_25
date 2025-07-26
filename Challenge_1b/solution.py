import os
import json
import pdfplumber
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Small, fast, good for semantic search

def extract_sections_and_paragraphs(pdf_path):
    sections = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(extra_attrs=["size"])
            if not words:
                continue
            # Find the largest font size on the page (likely section titles)
            max_size = max(w['size'] for w in words)
            # Group words by their vertical position (lines)
            lines = {}
            for w in words:
                y = round(w['top'], 1)
                lines.setdefault(y, []).append((w['text'], w['size']))
            # Identify section titles and paragraphs
            sorted_lines = sorted(lines.items())
            current_title = None
            current_paragraph = []
            for y, line_words in sorted_lines:
                line_text = " ".join([w[0] for w in line_words]).strip()
                line_size = max([w[1] for w in line_words])
                if line_size == max_size and len(line_text) > 5:
                    # Save previous section if exists
                    if current_title and current_paragraph:
                        sections.append({
                            "section_title": current_title,
                            "refined_text": " ".join(current_paragraph).strip(),
                            "page_number": page_num
                        })
                    current_title = line_text
                    current_paragraph = []
                else:
                    if current_title:
                        current_paragraph.append(line_text)
            # Save last section on page
            if current_title and current_paragraph:
                sections.append({
                    "section_title": current_title,
                    "refined_text": " ".join(current_paragraph).strip(),
                    "page_number": page_num
                })
    return sections

def process_collection(collection_path):
    input_json = os.path.join(collection_path, "challenge1b_input.json")
    output_json = os.path.join(collection_path, "challenge1b_output.json")
    pdf_dir = os.path.join(collection_path, "PDFs")

    if not os.path.exists(input_json):
        print(f"Input JSON not found in {collection_path}, skipping.")
        return

    with open(input_json, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    persona = input_data["persona"]["role"]
    job = input_data["job_to_be_done"]["task"]
    documents = input_data["documents"]

    # Load embedding model once per collection
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Parse PDFs into sections
    all_sections = []
    for doc in tqdm(documents, desc=f"Parsing PDFs in {collection_path}"):
        pdf_path = os.path.join(pdf_dir, doc["filename"])
        if not os.path.exists(pdf_path):
            print(f"Warning: {pdf_path} not found, skipping.")
            continue
        sections = extract_sections_and_paragraphs(pdf_path)
        for sec in sections:
            all_sections.append({
                "document": doc["filename"],
                "section_title": sec["section_title"],
                "refined_text": sec["refined_text"],
                "page_number": sec["page_number"]
            })

    if not all_sections:
        print(f"No sections found in {collection_path}, skipping output.")
        return

    # Embed refined texts
    section_texts = [s["refined_text"] for s in all_sections]
    section_embeddings = model.encode(section_texts, show_progress_bar=True, convert_to_numpy=True)

    # Embed persona+job
    persona_job_text = f"{persona}. {job}"
    persona_embedding = model.encode([persona_job_text], convert_to_numpy=True)[0]

    # Cosine similarity
    def cosine_similarity(a, b):
        a = a / (np.linalg.norm(a) + 1e-10)
        b = b / (np.linalg.norm(b) + 1e-10)
        return np.dot(a, b)

    similarities = np.array([cosine_similarity(persona_embedding, emb) for emb in section_embeddings])

    # Rank sections
    top_indices = similarities.argsort()[::-1]
    top_n = 5  # or any number you want

    # Format output
    output = {
        "metadata": {
            "input_documents": [doc["filename"] for doc in documents],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": __import__("datetime").datetime.now().isoformat()
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }

    for rank, idx in enumerate(top_indices[:top_n], start=1):
        sec = all_sections[idx]
        output["extracted_sections"].append({
            "document": sec["document"],
            "section_title": sec["section_title"],
            "importance_rank": rank,
            "page_number": sec["page_number"]
        })
        output["subsection_analysis"].append({
            "document": sec["document"],
            "refined_text": sec["refined_text"],
            "page_number": sec["page_number"]
        })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"Output written to {output_json}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    # Find all collections (folders starting with "Collection")
    for collection in sorted(base_dir.glob("Collection*")):
        if collection.is_dir():
            print(f"\nProcessing {collection} ...")
            process_collection(str(collection))