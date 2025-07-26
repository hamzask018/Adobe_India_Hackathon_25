import os
import json
import pdfplumber
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Small, fast, good for semantic search

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

    # Load embedding model once per collection (could be moved outside for speed)
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Parse PDFs into paragraphs
    all_paragraphs = []
    for doc in tqdm(documents, desc=f"Parsing PDFs in {collection_path}"):
        pdf_path = os.path.join(pdf_dir, doc["filename"])
        if not os.path.exists(pdf_path):
            print(f"Warning: {pdf_path} not found, skipping.")
            continue
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue
                # Split into paragraphs (double newline or single newline fallback)
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                if len(paragraphs) == 1:
                    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
                for para in paragraphs:
                    all_paragraphs.append({
                        "document": doc["filename"],
                        "section_title": para[:80],
                        "full_text": para,
                        "page_number": page_num
                    })

    if not all_paragraphs:
        print(f"No paragraphs found in {collection_path}, skipping output.")
        return

    # Embed paragraphs
    paragraph_texts = [p["full_text"] for p in all_paragraphs]
    paragraph_embeddings = model.encode(paragraph_texts, show_progress_bar=True, convert_to_numpy=True)

    # Embed persona+job
    persona_job_text = f"{persona}. {job}"
    persona_embedding = model.encode([persona_job_text], convert_to_numpy=True)[0]

    # Cosine similarity
    def cosine_similarity(a, b):
        a = a / (np.linalg.norm(a) + 1e-10)
        b = b / (np.linalg.norm(b) + 1e-10)
        return np.dot(a, b)

    similarities = np.array([cosine_similarity(persona_embedding, emb) for emb in paragraph_embeddings])

    # Rank paragraphs
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
        para = all_paragraphs[idx]
        output["extracted_sections"].append({
            "document": para["document"],
            "section_title": para["section_title"],
            "importance_rank": rank,
            "page_number": para["page_number"]
        })
        output["subsection_analysis"].append({
            "document": para["document"],
            "refined_text": para["full_text"],
            "page_number": para["page_number"]
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