import os
import fitz
import json
from collections import Counter

INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"

MAX_HEADING_WORDS = 16

def pdf_to_markdown_and_outline(pdf_path):
    doc = fitz.open(pdf_path)
    all_spans = []

    for page_idx, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                merged_text = " ".join([span["text"].strip() for span in line["spans"] if span["text"].strip()])
                if merged_text:
                    line_span = {
                        "text": merged_text,
                        "size": max(span["size"] for span in line["spans"] if span["text"].strip()),
                        "_line_top": line["bbox"][1],
                        "_line_bottom": line["bbox"][3],
                        "_page_number": page_idx + 1,
                        "word_count": len(merged_text.split())
                    }
                    all_spans.append(line_span)

    font_sizes = [span["size"] for span in all_spans]
    if font_sizes:
        body_size = Counter(font_sizes).most_common(1)[0][0]
        heading_sizes = sorted(set(s for s in font_sizes if s > body_size), reverse=True)
        size_to_level = {size: idx+1 for idx, size in enumerate(heading_sizes[:6])}
    else:
        body_size = 0
        size_to_level = {}

    def is_heading_size(size):
        return size > body_size

    outline = []
    title = ""
    markdown_lines = []

    i = 0
    n = len(all_spans)
    while i < n:
        curr = all_spans[i]
        group = [curr]
        group_font_sizes = [curr["size"]]
        group_texts = [curr["text"]]
        group_word_counts = [curr["word_count"]]
        page_num = curr["_page_number"]
        max_bottom = curr["_line_bottom"]

        j = i + 1
        while j < n:
            next_span = all_spans[j]
            if next_span["_page_number"] != page_num:
                break
            vertical_gap = next_span["_line_top"] - max_bottom
            if 0 <= vertical_gap < 1.5 * body_size:
                group.append(next_span)
                group_font_sizes.append(next_span["size"])
                group_texts.append(next_span["text"])
                group_word_counts.append(next_span["word_count"])
                max_bottom = max(max_bottom, next_span["_line_bottom"])
                j += 1
            else:
                break

        has_heading = any(is_heading_size(sz) for sz in group_font_sizes)
        total_words = sum(group_word_counts)
        if has_heading and total_words <= MAX_HEADING_WORDS:
            max_font = max(group_font_sizes)
            group_level = size_to_level.get(max_font)
            heading_text = " ".join(group_texts)
            if group_level:
                markdown_lines.append(f"{'#' * group_level} {heading_text}\n")
                if not title:
                    title = heading_text
                outline.append({
                    "level": f"H{group_level}",
                    "text": heading_text,
                    "page": page_num
                })
                i = j
                continue
            
        for text in group_texts:
            markdown_lines.append(text + "\n")
        i = j

    markdown_content = "\n".join(markdown_lines)
    return title, outline, markdown_content

def process_pdf(pdf_filename):
    pdf_path = os.path.join(INPUT_DIR, pdf_filename)
    output_path = os.path.join(OUTPUT_DIR, os.path.splitext(pdf_filename)[0] + ".json")
    title, outline, _ = pdf_to_markdown_and_outline(pdf_path)
    result = {
        "title": title or os.path.splitext(pdf_filename)[0],
        "outline": outline
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(".pdf"):
            print(f"Processing: {filename}")
            try:
                process_pdf(filename)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
    print("Done.")

if __name__ == "__main__":
    main()