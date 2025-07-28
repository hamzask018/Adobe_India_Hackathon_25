# Adobe_India_Hackathon_25

This repo is for the submission of the work done for the Adobe India Hackathon 2025.

---

## Challenge 1b: Multi-Collection PDF Analysis

This solution processes multiple PDF collections, extracts and ranks relevant sections based on a persona and job-to-be-done, and outputs structured JSON results. The approach combines machine learning and heuristics for robust section title detection and semantic ranking.

---

## Setup & Running

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Train the section title model:**
   ```sh
   python model.py
   ```
   This will process all collections, extract features and labels from PDFs and output JSONs, and save `section_title_model.pkl`.

3. **Run the extraction and ranking solution:**
   ```sh
   python solution.py
   ```
   This will process each collection, extract and summarize sections, rank them, and write the results to `challenge1b_output.json` in each collection folder.

---

## Approach

### model.py

**Purpose:**  
Trains a machine learning model (RandomForestClassifier) to detect section titles in PDFs.

**How it works:**
- Iterates through all collections and their PDFs.
- For each line in each PDF, extracts features:
  - Font size
  - Is the line all uppercase
  - Length of the line
  - Is the first character capitalized
- Uses the provided `challenge1b_output.json` files to label lines as section titles or not.
- Builds a training dataset from these features and labels.
- Trains a RandomForestClassifier on this dataset.
- Saves the trained model as `section_title_model.pkl` for use in extraction.

**Why this approach:**  
RandomForest is robust for tabular data and can handle the variety of formatting found in different PDFs. Training on real labeled data from your outputs ensures the model learns relevant patterns.

---

### solution.py

**Purpose:**  
Extracts, summarizes, and ranks relevant sections from PDFs using the trained model, and writes the results to output JSON files.

**How it works:**
- For each collection, loads the trained section title model and a transformer-based summarizer.
- For each PDF:
  - Uses the ML model to detect section titles based on extracted features.
  - Groups following lines as the section's paragraph.
  - Summarizes each section's paragraph using a transformer summarizer (`facebook/bart-large-cnn`).
  - If the ML model finds no sections, falls back to a heuristic (largest font size per page).
- Embeds each section's summary and the persona/job description using SentenceTransformer.
- Ranks sections by semantic similarity to the persona/job.
- Writes a structured output JSON with:
  - Metadata (input docs, persona, job, timestamp)
  - Top-ranked extracted sections
  - Subsection analysis (summaries)
- **Always writes an output file**, even if no sections are found (fields will be empty but metadata is included).

**Why this approach:**  
Combining ML and heuristics ensures robust section detection across diverse document formats. Semantic ranking tailors the output to the user's needs, and automatic summarization provides concise, relevant content.

---

## Project Structure

```
Challenge_1b/
├── model.py                  # Training script for section title classifier
├── solution.py               # Main extraction, ranking, and output script
├── requirements.txt          # All dependencies
├── section_title_model.pkl   # Trained ML model (created by model.py)
├── Collection 1/
│   ├── PDFs/
│   ├── challenge1b_input.json
│   └── challenge1b_output.json
├── Collection 2/
│   ├── PDFs/
│   ├── challenge1b_input.json
│   └── challenge1b_output.json
├── Collection 3/
│   ├── PDFs/
│   ├── challenge1b_input.json
│   └── challenge1b_output.json
└── README.md
```

---

## Input/Output Format

### Input (`challenge1b_input.json`):
```json
{
  "challenge_info": { ... },
  "documents": [{"filename": "doc.pdf", "title": "Title"}],
  "persona": {"role": "User Persona"},
  "job_to_be_done": {"task": "Use case description"}
}
```

### Output (`challenge1b_output.json`):
```json
{
  "metadata": {
    "input_documents": ["list of PDFs"],
    "persona": "User Persona",
    "job_to_be_done": "Task description",
    "processing_timestamp": "..."
  },
  "extracted_sections": [
    {
      "document": "source.pdf",
      "section_title": "Title",
      "importance_rank": 1,
      "page_number": 1
    }
  ],
  "subsection_analysis": [
    {
      "document": "source.pdf",
      "refined_text": "Content",
      "page_number": 1
    }
  ]
}
```
If no sections are found, `extracted_sections` and `subsection_analysis` will be empty, but metadata will always be present.

---

## Notes

- For best results, ensure your PDFs are text-based (not scanned images).
- You can retrain the model at any time by running `model.py` if you add new labeled data or collections.
- The solution is robust and will always produce an output JSON for each collection.

---
