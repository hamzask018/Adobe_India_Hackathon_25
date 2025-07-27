import os
import json
import pdfplumber
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import joblib
from transformers import pipeline
import pandas as pd

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TITLE_MODEL_PATH = "section_title_model.pkl"

def extract_features(line_text, line_size):
    return [
        line_size,
        int(line_text.isupper()),
        len(line_text),
        int(line_text and line_text[0].isupper())
    ]

def extract_sections_ml(pdf_path, clf, summarizer):
    sections = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(extra_attrs=["size"])
            if not words:
                continue
            lines = {}
            for w in words:
                y = round(w['top'], 1)
                lines.setdefault(y, []).append(w)
            sorted_lines = sorted(lines.items())
            current_title = None
            current_paragraph = []
            found_title = False
            for y, line_words in sorted_lines:
                line_text = " ".join([w['text'] for w in line_words]).strip()
                line_size = max([w['size'] for w in line_words])
                features = [line_size, int(line_text.isupper()), len(line_text), int(line_text and line_text[0].isupper())]
                df_features = pd.DataFrame([features], columns=["size", "is_upper", "length", "is_capitalized"])
                is_title = clf.predict(df_features)[0]
                if is_title:
                    found_title = True
                    if current_title and current_paragraph:
                        full_text = " ".join(current_paragraph).strip()
                        summary = summarizer(full_text, max_length=120, min_length=30, do_sample=False)[0]['summary_text'] if full_text else ""
                        sections.append({
                            "section_title": current_title,
                            "refined_text": summary,
                            "page_number": page_num
                        })
                    current_title = line_text
                    current_paragraph = []
                else:
                    if current_title:
                        current_paragraph.append(line_text)
            if current_title and current_paragraph:
                full_text = " ".join(current_paragraph).strip()
                summary = summarizer(full_text, max_length=120, min_length=30, do_sample=False)[0]['summary_text'] if full_text else ""
                sections.append({
                    "section_title": current_title,
                    "refined_text": summary,
                    "page_number": page_num
                })
    return sections

def extract_sections_heuristic(pdf_path, summarizer):
    # Fallback: use largest font size per page as section title
    sections = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(extra_attrs=["size"])
            if not words:
                continue
            max_size = max(w['size'] for w in words)
            lines = {}
            for w in words:
                y = round(w['top'], 1)
                lines.setdefault(y, []).append(w)
            sorted_lines = sorted(lines.items())
            current_title = None
            current_paragraph = []
            for y, line_words in sorted_lines:
                line_text = " ".join([w['text'] for w in line_words]).strip()
                line_size = max([w['size'] for w in line_words])
                if line_size == max_size and len(line_text) > 5:
                    if current_title and current_paragraph:
                        full_text = " ".join(current_paragraph).strip()
                        summary = summarizer(full_text, max_length=120, min_length=30, do_sample=False)[0]['summary_text'] if full_text else ""
                        sections.append({
                            "section_title": current_title,
                            "refined_text": summary,
                            "page_number": page_num
                        })
                    current_title = line_text
                    current_paragraph = []
                else:
                    if current_title:
                        current_paragraph.append(line_text)
            if current_title and current_paragraph:
                full_text = " ".join(current_paragraph).strip()
                summary = summarizer(full_text, max_length=120, min_length=30, do_sample=False)[0]['summary_text'] if full_text else ""
                sections.append({
                    "section_title": current_title,
                    "refined_text": summary,
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

    model = SentenceTransformer(EMBEDDING_MODEL)
    clf = joblib.load(TITLE_MODEL_PATH)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    all_sections = []
    for doc in tqdm(documents, desc=f"Parsing PDFs in {collection_path}"):
        pdf_path = os.path.join(pdf_dir, doc["filename"])
        if not os.path.exists(pdf_path):
            print(f"Warning: {pdf_path} not found, skipping.")
            continue
        sections = extract_sections_ml(pdf_path, clf, summarizer)
        # Fallback to heuristic if ML finds no sections
        if not sections:
            print(f"ML found no sections in {doc['filename']}, using heuristic.")
            sections = extract_sections_heuristic(pdf_path, summarizer)
        for sec in sections:
            all_sections.append({
                "document": doc["filename"],
                "section_title": sec["section_title"],
                "refined_text": sec["refined_text"],
                "page_number": sec["page_number"]
            })

    # Always write output file, even if empty
    section_texts = [s["refined_text"] for s in all_sections]
    if section_texts:
        section_embeddings = model.encode(section_texts, show_progress_bar=True, convert_to_numpy=True)
        persona_job_text = f"{persona}. {job}"
        persona_embedding = model.encode([persona_job_text], convert_to_numpy=True)[0]

        def cosine_similarity(a, b):
            a = a / (np.linalg.norm(a) + 1e-10)
            b = b / (np.linalg.norm(b) + 1e-10)
            return np.dot(a, b)

        similarities = np.array([cosine_similarity(persona_embedding, emb) for emb in section_embeddings])
        top_indices = similarities.argsort()[::-1]
        top_n = min(5, len(all_sections))

        extracted_sections = []
        subsection_analysis = []
        for rank, idx in enumerate(top_indices[:top_n], start=1):
            sec = all_sections[idx]
            extracted_sections.append({
                "document": sec["document"],
                "section_title": sec["section_title"],
                "importance_rank": rank,
                "page_number": sec["page_number"]
            })
            subsection_analysis.append({
                "document": sec["document"],
                "refined_text": sec["refined_text"],
                "page_number": sec["page_number"]
            })
    else:
        extracted_sections = []
        subsection_analysis = []

    output = {
        "metadata": {
            "input_documents": [doc["filename"] for doc in documents],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": __import__("datetime").datetime.now().isoformat()
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"Output written to {output_json}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    for collection in sorted(base_dir.glob("Collection*")):
        if collection.is_dir():
            print(f"\nProcessing {collection} ...")
            process_collection(str(collection))