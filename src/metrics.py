"""Exact entity-span evaluation and error extraction."""
from collections import defaultdict
from .config import ENTITY_TYPES


def entity_set(entities):
    return {(e["start"], e["end"], e["label"]) for e in entities}


def safe_prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "support": tp + fn}


def evaluate_predictions(gold_docs, prediction_lists):
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    errors = []
    for doc, predicted in zip(gold_docs, prediction_lists):
        gold = entity_set(doc["entities"])
        pred = entity_set(predicted)
        for label in ENTITY_TYPES:
            g = {x for x in gold if x[2] == label}
            p = {x for x in pred if x[2] == label}
            counts[label]["tp"] += len(g & p)
            counts[label]["fp"] += len(p - g)
            counts[label]["fn"] += len(g - p)
        for start, end, label in sorted(pred - gold):
            errors.append({
                "doc_id": doc["doc_id"], "error": "FALSE_POSITIVE",
                "text": doc["text"][start:end], "label": label,
                "start": start, "end": end
            })
        for start, end, label in sorted(gold - pred):
            errors.append({
                "doc_id": doc["doc_id"], "error": "FALSE_NEGATIVE",
                "text": doc["text"][start:end], "label": label,
                "start": start, "end": end
            })

    per_class = {}
    total_tp = total_fp = total_fn = 0
    class_f1 = []
    for label in ENTITY_TYPES:
        c = counts[label]
        per_class[label] = safe_prf(c["tp"], c["fp"], c["fn"])
        total_tp += c["tp"]; total_fp += c["fp"]; total_fn += c["fn"]
        class_f1.append(per_class[label]["f1"])
    micro = safe_prf(total_tp, total_fp, total_fn)
    macro = sum(class_f1) / len(class_f1) if class_f1 else 0.0
    return {
        "per_class": per_class,
        "micro": micro,
        "macro_f1": macro,
        "counts": {"tp": total_tp, "fp": total_fp, "fn": total_fn},
    }, errors
