"""Feature extraction for a linear-chain CRF."""
def token_features(tokens, i):
    token = tokens[i]["text"]
    lower = token.lower()
    features = {
        "bias": 1.0,
        "word.lower": lower,
        "word": token,
        "word.isupper": token.isupper(),
        "word.istitle": token.istitle(),
        "word.islower": token.islower(),
        "word.isdigit": token.isdigit(),
        "word.hasdigit": any(c.isdigit() for c in token),
        "word.has_hyphen": "-" in token,
        "word.has_punct": any(not c.isalnum() for c in token),
        "word.prefix1": lower[:1],
        "word.prefix2": lower[:2],
        "word.prefix3": lower[:3],
        "word.suffix1": lower[-1:],
        "word.suffix2": lower[-2:],
        "word.suffix3": lower[-3:],
        "position.is_first": i == 0,
        "position.is_last": i == len(tokens) - 1,
    }
    if i > 0:
        prev = tokens[i - 1]["text"]
        features.update({
            "-1:word.lower": prev.lower(),
            "-1:word.istitle": prev.istitle(),
            "-1:word.isupper": prev.isupper(),
            "-1:word.isdigit": prev.isdigit(),
        })
    else:
        features["BOS"] = True
    if i < len(tokens) - 1:
        nxt = tokens[i + 1]["text"]
        features.update({
            "+1:word.lower": nxt.lower(),
            "+1:word.istitle": nxt.istitle(),
            "+1:word.isupper": nxt.isupper(),
            "+1:word.isdigit": nxt.isdigit(),
        })
    else:
        features["EOS"] = True
    return features


def document_features(tokens):
    return [token_features(tokens, i) for i in range(len(tokens))]
