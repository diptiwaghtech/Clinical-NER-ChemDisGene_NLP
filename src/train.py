"""Prepare data, train baseline and CRF, save model artifacts."""
import json
from .data import prepare_data, load_split
from .baseline import DictionaryBaseline
from .modeling import train_crf
from .config import MODEL_DIR, REPORT_DIR


def main():
    prepare_data()
    train_docs = load_split("train")
    valid_docs = load_split("validation")

    baseline = DictionaryBaseline().fit(train_docs)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(baseline, MODEL_DIR / "dictionary_baseline.joblib")

    model = train_crf(train_docs)
    # Validation inference is saved as a sanity-check artifact; final metrics use test only.
    validation_examples = []
    from .modeling import predict_crf
    for doc in valid_docs[:10]:
        validation_examples.append({
            "doc_id": doc["doc_id"],
            "text": doc["text"],
            "gold": doc["entities"],
            "predicted": predict_crf(model, doc["text"]),
        })
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "validation_examples.json").write_text(
        json.dumps(validation_examples, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Trained CRF on {len(train_docs)} documents.")
    print(f"Validation documents available: {len(valid_docs)}")
    print(f"Saved model to {MODEL_DIR / 'crf_model.joblib'}")


if __name__ == "__main__":
    main()
