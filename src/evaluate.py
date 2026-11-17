"""Evaluate dictionary and CRF on held-out test documents."""
import csv
import json
from .data import load_split
from .baseline import DictionaryBaseline
from .modeling import load_crf, predict_crf
from .metrics import evaluate_predictions
from .config import MODEL_DIR, REPORT_DIR
import joblib


def main():
    test_docs = load_split("test")
    baseline = joblib.load(MODEL_DIR / "dictionary_baseline.joblib")
    crf = load_crf()

    outputs = {}
    all_errors = []
    for name, predictor in [
        ("dictionary_baseline", lambda d: baseline.predict_document(d["text"])),
        ("crf", lambda d: predict_crf(crf, d["text"])),
    ]:
        predictions = [predictor(doc) for doc in test_docs]
        metrics, errors = evaluate_predictions(test_docs, predictions)
        outputs[name] = metrics
        for err in errors:
            err["model"] = name
        all_errors.extend(errors)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(outputs, indent=2), encoding="utf-8"
    )
    with (REPORT_DIR / "error_analysis.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["model", "doc_id", "error", "text", "label", "start", "end"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_errors)

    # Save representative CRF outputs for inspection.
    sample_rows = []
    for doc in test_docs[:10]:
        sample_rows.append({
            "doc_id": doc["doc_id"],
            "text": doc["text"],
            "gold_entities": doc["entities"],
            "predicted_entities": predict_crf(crf, doc["text"]),
        })
    (REPORT_DIR / "sample_predictions.json").write_text(
        json.dumps(sample_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = ["Exact entity-span evaluation (end-exclusive offsets)", ""]
    for model_name, result in outputs.items():
        lines.append(model_name)
        for label, scores in result["per_class"].items():
            lines.append(
                f"  {label}: precision={scores['precision']:.4f}, "
                f"recall={scores['recall']:.4f}, F1={scores['f1']:.4f}, "
                f"support={scores['support']}"
            )
        lines.append(
            f"  MICRO: precision={result['micro']['precision']:.4f}, "
            f"recall={result['micro']['recall']:.4f}, "
            f"F1={result['micro']['f1']:.4f}"
        )
        lines.append(f"  MACRO F1={result['macro_f1']:.4f}")
        lines.append("")
    (REPORT_DIR / "classification_report.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("\n".join(lines))
    print(f"Reports saved in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
