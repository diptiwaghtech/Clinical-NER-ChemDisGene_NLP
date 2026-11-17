# Clinical Named Entity Recognition with ChemDisGene

**Classical NLP | CRF | Python | Hugging Face dataset**

An end-to-end three-class biomedical NER project that detects:

- `CHEMICAL` — chemical substances and compounds
- `DISEASE` — diseases and disorders
- `GENE` — genes and gene products

This project is based on the assessment brief `Clinical_NER_Candidate_Assessment.pdf`. It includes a dictionary baseline, a trained Conditional Random Field (CRF), entity-level evaluation, error analysis, and reusable inference returning original-text character offsets.

> **Scope note:** ChemDisGene contains biomedical research abstracts, not hospital/clinical notes. It is used here as an openly accessible biomedical NER dataset to demonstrate the requested pipeline. Results should not be represented as validated clinical-note performance.

## Dataset

- Hugging Face: <https://huggingface.co/datasets/bigbio/chem_dis_gene>
- Original project: <https://github.com/chanzuckerberg/ChemDisGene>
- Paper: Zhang et al. (2022), *A Distant Supervision Corpus for Extracting Biomedical Relationships Between Chemicals, Diseases and Genes*, LREC 2022.
- The Hugging Face dataset script's curated configuration downloads the curated PubTator abstracts. The script defines `chem_dis_gene_source` and `chem_dis_gene_bigbio_kb` configurations, and the curated dataset is exposed as one split.
- This implementation downloads the same curated `abstracts.txt.gz` source file from the original ChemDisGene repository and parses its PubTator annotations. This avoids dependence on legacy Hugging Face dataset-script execution.

The dataset card reports CC0-1.0 for the dataset script. Check the upstream repository and applicable terms before redistribution.

## Project workflow

1. Download and parse the curated PubTator abstracts.
2. Join title and abstract with one space; retain annotation character offsets.
3. Tokenize text with a regex tokenizer that records original character positions.
4. Convert character spans to BIO tags (`B-CHEMICAL`, `I-CHEMICAL`, etc.).
5. Create reproducible document-level 70% train / 15% validation / 15% test partitions.
6. Train a dictionary-matching baseline from training annotations.
7. Train a CRF using lexical, orthographic, affix and neighboring-token features.
8. Evaluate exact entity spans with per-class precision, recall and F1.
9. Run error analysis and save example predictions.
10. Use the inference function to return entity text, label, start and end offsets.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

## Run the complete pipeline

```bash
python -m src.train
python -m src.evaluate
```

The first command downloads the dataset if it is not already present, prepares splits, trains the baseline and CRF, and saves the CRF model and metadata under `models/`.

The second command evaluates both systems on the held-out test set and writes:
- `reports/metrics.json`
- `reports/classification_report.txt`
- `reports/error_analysis.csv`
- `reports/sample_predictions.json`

Run inference on a new note:

```bash
python -m src.predict --text "Aspirin was given to a patient with diabetes. EGFR was measured."
```

Or use Python:

```python
from src.predict import predict_entities

entities = predict_entities(
    "Aspirin was given to a patient with diabetes.",
    model_path="models/crf_model.joblib"
)
print(entities)
```

Output format:

```json
[
  {
    "text": "Aspirin",
    "label": "CHEMICAL",
    "start": 0,
    "end": 7
  }
]
```

Offsets are zero-based, end-exclusive character offsets into the exact input string.

## Model details

### Baseline
A case-insensitive longest-match dictionary built only from entity mentions in the training partition. It is a simple, transparent baseline; it cannot reliably detect unseen names or resolve ambiguous terms from context.

### CRF
A linear-chain Conditional Random Field trained with `sklearn-crfsuite`. Features include:
- Lowercased token and original token
- Prefixes and suffixes
- Capitalization, digits, punctuation and token shape
- Previous and next token features
- Beginning/end-of-sequence indicators

The CRF predicts a BIO tag sequence and uses transition information to encourage coherent entity spans.

## Evaluation

The evaluation script uses exact span matching: a prediction is correct only when its start, end and entity class all match a gold entity. It reports entity-level precision, recall and F1 for each class, plus micro and macro averages. It also exports false positives and false negatives for manual inspection.

Do not compare token accuracy with entity-level F1 as if they were the same metric. The primary metric is exact entity-level F1.

## Reproducibility

- Random seed: `42`
- Split unit: document (not token or sentence)
- Split proportions: 70% / 15% / 15%
- The split manifest is saved to `data/processed/split_manifest.json`.
- Dataset is downloaded from the upstream public repository on first run.
- Model hyperparameters are in `src/config.py`.

Because the curated corpus is relatively small, metrics may vary with library versions and corpus parsing. Inspect the saved split manifest and reports when reproducing results.

## Limitations and responsible use

- Biomedical abstracts are not equivalent to real clinical notes.
- The curated collection is small compared with the full CTD-derived corpus.
- Chemical, disease and gene mentions may be ambiguous; dictionary matching is context-insensitive.
- Annotation coverage and source conventions can affect measured performance.
- CRF features are hand-engineered and may not generalize to unseen terminology.
- This is an educational NLP prototype, not a medical device or clinical decision-support system. Do not use its predictions for patient care.

## Repository structure

```text
clinical_ner_chemdisgene/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── baseline.py
│   ├── modeling.py
│   ├── metrics.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
├── sample_input.txt
└── PROJECT_REPORT.md
```

## Citation

Zhang, D., Mohan, S., Torkar, M., & McCallum, A. (2022). A Distant Supervision Corpus for Extracting Biomedical Relationships Between Chemicals, Diseases and Genes. *Proceedings of LREC 2022*, 1073–1082. https://aclanthology.org/2022.lrec-1.116/

## Assessment mapping

| Assessment requirement | Implementation |
|---|---|
| Data loading and annotations | `src/data.py` |
| Cleaning and preprocessing | `src/data.py` |
| Feature engineering | `src/features.py` |
| Baseline and trained model | `src/baseline.py`, `src/modeling.py` |
| Model training | `src/train.py` |
| Entity-level evaluation | `src/metrics.py`, `src/evaluate.py` |
| Error analysis | `reports/error_analysis.csv` |
| Inference and character offsets | `src/predict.py` |
