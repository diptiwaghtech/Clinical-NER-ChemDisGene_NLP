"""Training-only dictionary baseline with greedy longest matching."""
from collections import defaultdict
from .data import tokenize_with_offsets


class DictionaryBaseline:
    def __init__(self):
        self.entries = defaultdict(set)
        self.max_len = 1

    def fit(self, documents):
        for doc in documents:
            for ent in doc["entities"]:
                phrase = tuple(t["text"].lower() for t in tokenize_with_offsets(ent["text"]))
                if phrase:
                    self.entries[phrase].add(ent["label"])
                    self.max_len = max(self.max_len, len(phrase))
        # Resolve conflicting labels deterministically; ambiguity remains a limitation.
        self.lookup = {
            phrase: sorted(labels)[0] for phrase, labels in self.entries.items()
        }
        return self

    def predict_document(self, text):
        tokens = tokenize_with_offsets(text)
        predicted = []
        i = 0
        while i < len(tokens):
            match = None
            for size in range(min(self.max_len, len(tokens) - i), 0, -1):
                phrase = tuple(t["text"].lower() for t in tokens[i:i + size])
                if phrase in self.lookup:
                    match = (size, self.lookup[phrase])
                    break
            if match:
                size, label = match
                predicted.append({
                    "text": text[tokens[i]["start"]:tokens[i + size - 1]["end"]],
                    "label": label,
                    "start": tokens[i]["start"],
                    "end": tokens[i + size - 1]["end"],
                })
                i += size
            else:
                i += 1
        return predicted
