# Clinical Named Entity Recognition with ChemDisGene

**NLP | CRF | Python | Hugging Face dataset**

An end-to-end three-class biomedical NER project that detects:

- `CHEMICAL` — chemical substances and compounds
- `DISEASE` — diseases and disorders
- `GENE` — genes and gene products

## Dataset

- Hugging Face: <https://huggingface.co/datasets/bigbio/chem_dis_gene>
- Original project: <https://github.com/chanzuckerberg/ChemDisGene>
- Paper: Zhang et al. (2022), *A Distant Supervision Corpus for Extracting Biomedical Relationships Between Chemicals, Diseases and Genes*, LREC 2022.
- The Hugging Face dataset script's curated configuration downloads the curated PubTator abstracts.
- This implementation downloads the same curated `abstracts.txt.gz` source file from the original ChemDisGene repository and parses its PubTator annotations.


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

Python 3.10 or 3.11

```bash
python -m venv .venv
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

The model is evaluated using exact entity matching. A prediction is correct only if its start position, end position, and entity label match the actual entity.

We calculate precision, recall, and F1-score for each entity class, along with micro and macro averages. We also save incorrect predictions (false positives and false negatives) for further analysis.

The main performance metric is entity-level F1-score. 

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
- This is an educational NLP prototype, not a medical device or clinical decision-support system.

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
