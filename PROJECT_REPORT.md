# Project Report: Three-Class Biomedical Named Entity Recognition

## 1. Objective
Build an end-to-end NLP system that identifies and classifies chemical, disease and gene/gene-product mentions in biomedical text. The work follows the Clinical NER Assessment and demonstrates data preparation, feature engineering, a meaningful baseline, a trained model, evaluation, error analysis and reusable inference.

## 2. Dataset and task framing
The project uses the curated ChemDisGene PubTator abstracts distributed through the Hugging Face `bigbio/chem_dis_gene` dataset. It includes mention-level annotations for chemical, disease and gene/gene-product entities. The selected source is biomedical abstracts rather than actual patient clinical notes. Consequently, this is a biomedical NER demonstration aligned with the assessment's three entity classes, not evidence of clinical deployment readiness.

The upstream Hugging Face script describes a curated configuration and uses the PubTator abstracts source. This repository downloads that same curated abstracts file directly from the upstream repository and parses the PubTator annotations.

## 3. Entity definitions
| Class | Definition | Example (illustrative) |
|---|---|---|
| CHEMICAL | Chemical substance, compound or drug-like chemical mention | aspirin |
| DISEASE | Disease or disorder mention | diabetes |
| GENE | Gene or gene-product mention | EGFR |

Examples above illustrate label meaning; they are not claimed to be particular dataset records.

## 4. Preprocessing
- Load compressed PubTator text.
- Group title, abstract and annotation lines by PubMed document ID.
- Join title and abstract with a single space.
- Preserve source annotation offsets and verify the exact mention text at the span.
- Tokenize with a regex tokenizer while retaining token start/end character offsets.
- Convert entity spans to BIO sequence labels.
- Skip annotations whose offsets do not exactly align with token boundaries; this is preferable to silently shifting source annotations.

## 5. Data split
The curated source is exposed as a single split, so the project creates a deterministic document-level split using seed 42: 70% training, 15% validation and the remaining 15% test. Documents are kept intact to avoid putting sentences from the same abstract into different partitions. The actual document IDs and split sizes are written to `data/processed/split_manifest.json` when the pipeline runs.

## 6. Baseline
The dictionary baseline extracts entity surface forms from training annotations and applies case-insensitive longest phrase matching to the test text. It is transparent and easy to explain. Its weaknesses are unseen entity names, ambiguous abbreviations, context-insensitive matching and limited handling of spelling variants.

## 7. CRF model
A linear-chain Conditional Random Field predicts BIO tags. Each token is represented by lexical and orthographic features, including lowercased word, capitalization, digit and punctuation flags, prefixes, suffixes, and previous/next token context. The CRF learns both token-level feature weights and dependencies between adjacent labels.

## 8. Evaluation methodology
The primary evaluation is exact entity-span matching. A prediction counts as correct only if its start offset, end offset and entity class all match a gold annotation. The evaluation reports precision, recall and F1 for CHEMICAL, DISEASE and GENE, plus micro-averaged and macro F1.

- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 × Precision × Recall / (Precision + Recall)

The evaluation script also exports false positives and false negatives.

**Results:** Run `python -m src.train` followed by `python -m src.evaluate`. The actual measured metrics are written to `reports/metrics.json` and `reports/classification_report.txt`.

## 9. Error analysis questions
When reviewing `reports/error_analysis.csv`, investigate:
1. Are predictions missing because the entity was never seen in training?
2. Are abbreviations or gene symbols confused with ordinary words?
3. Are multi-token mentions predicted with incomplete boundaries?
4. Are chemical and gene names confused?
5. Does punctuation or hyphenation cause token-boundary problems?
6. Are there annotation inconsistencies in the source?

## 10. Limitations and next steps
- The corpus consists of biomedical abstracts, not de-identified hospital notes.
- Curated data is smaller than the full CTD-derived corpus.
- Some upstream annotations may be incomplete or ambiguous.
- A CRF relies on hand-designed features and may not generalize to new terminology.
- Dictionary matching is context-insensitive.
- The model is not suitable for diagnosis, treatment decisions or patient-care use.

Potential next steps include evaluating on a separately licensed clinical-note NER corpus, adding carefully reviewed synthetic clinical examples, improving abbreviation/context features, and conducting error analysis with a domain expert.

## 11. Reproducibility and artifacts
Use `requirements.txt`, seed 42, the saved split manifest, and the documented CLI commands in `README.md`. Model files and generated reports are created locally after training.
