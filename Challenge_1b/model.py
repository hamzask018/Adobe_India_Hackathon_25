import os
import json
import pdfplumber
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def extract_features(line_text, line_size):
    return {
        "size": line_size,
        "is_upper": int(line_text.isupper()),
        "length": len(line_text),
        "is_capitalized": int(line_text and line_text[0].isupper())
    }

def build_training_data(collections):
    rows = []
    for collection in collections:
        pdf_dir = os.path.join(collection, "PDFs")
        output_json_path = os.path.join(collection, "challenge1b_output.json")
        if not os.path.exists(output_json_path):
            print(f"Warning: {output_json_path} not found, skipping.")
            continue
        with open(output_json_path, "r", encoding="utf-8") as f:
            output_data = json.load(f)
        title_map = {}
        for sec in output_data.get("extracted_sections", []):
            title_map[(sec["document"], sec["section_title"], sec["page_number"])] = 1
        for pdf_file in os.listdir(pdf_dir):
            if not pdf_file.endswith(".pdf"):
                continue
            pdf_path = os.path.join(pdf_dir, pdf_file)
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
                    for y, line_words in sorted_lines:
                        line_text = " ".join([w['text'] for w in line_words]).strip()
                        line_size = max([w['size'] for w in line_words])
                        label = 1 if (pdf_file, line_text, page_num) in title_map else 0
                        feat = extract_features(line_text, line_size)
                        feat["label"] = label
                        rows.append(feat)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    # Use all collections for training
    base_dir = os.path.dirname(__file__)
    collections = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if d.startswith("Collection")]
    df = build_training_data(collections)
    if df.empty:
        print("No training data found. Please ensure output.json files exist.")
    else:
        X = df[["size", "is_upper", "length", "is_capitalized"]]
        y = df["label"]
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        joblib.dump(clf, "section_title_model.pkl")
        print("Model trained and saved as section_title_model.pkl")