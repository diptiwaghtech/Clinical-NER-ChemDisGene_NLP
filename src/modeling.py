"""CRF training and prediction helpers."""
import joblib
import sklearn_crfsuite
from .config import CRF_PARAMS, MODEL_DIR
from .data import tokenize_with_offsets
from .features import document_features


def train_crf(documents):
    X, y = [], []
    from .data import document_to_sequences
    for doc in documents:
        tokens, tags = document_to_sequences(doc)
        if tokens:
            X.append(document_features(tokens))
            y.append(tags)
    if not X:
        raise ValueError("No token sequences available for CRF training.")
    model = sklearn_crfsuite.CRF(**CRF_PARAMS)
    model.fit(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "crf_model.joblib")
    return model


def load_crf(path=None):
    return joblib.load(path or (MODEL_DIR / "crf_model.joblib"))


def tags_to_entities(text, tokens, tags):
    entities = []
    current = None
    for token, tag in zip(tokens, tags):
        if tag == "O" or "-" not in tag:
            if current:
                entities.append(current)
                current = None
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "B" or current is None or current["label"] != label:
            if current:
                entities.append(current)
            current = {
                "text": text[token["start"]:token["end"]],
                "label": label,
                "start": token["start"],
                "end": token["end"],
            }
        else:
            current["end"] = token["end"]
            current["text"] = text[current["start"]:current["end"]]
    if current:
        entities.append(current)
    return entities


def predict_crf(model, text):
    tokens = tokenize_with_offsets(text)
    if not tokens:
        return []
    tags = model.predict_single(document_features(tokens))
    return tags_to_entities(text, tokens, tags)
