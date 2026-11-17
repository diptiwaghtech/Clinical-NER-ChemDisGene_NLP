"""Reusable inference API and command-line prediction."""
import argparse
import json
from .modeling import load_crf, predict_crf


def predict_entities(text, model_path="models/crf_model.joblib"):
    """Return entity text, class and original-text character offsets."""
    model = load_crf(model_path)
    return predict_crf(model, text)


def main():
    parser = argparse.ArgumentParser(description="Three-class ChemDisGene NER inference")
    parser.add_argument("--text", required=True, help="Input biomedical text")
    parser.add_argument("--model", default="models/crf_model.joblib")
    args = parser.parse_args()
    print(json.dumps(predict_entities(args.text, args.model),
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
