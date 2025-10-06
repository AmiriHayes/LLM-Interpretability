import numpy as np
from transformers import PreTrainedTokenizerBase


def punctuation_conjunction_dependency(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    input_ids = toks.input_ids[0].numpy()
    token_sequence = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True)

    # Dictionary mapping some common punctuation and conjunction symbols for easy reference
    special_tokens = {",": "comma", ".": "period", "?": "question", "!": "exclamation",
                      "and": "and", "or": "or", "but": "but"}

    identified_positions = {}
    for i, token in enumerate(token_sequence, 1):
        if token in special_tokens:
            identified_positions[i] = special_tokens[token]

    # Punctuation tends to have higher weights as seen in the data
    for pos in identified_positions:
        # Assign non-zero attention weights for identified punctuation positions
        out[pos] = 1.0
        out[:, pos] = 1.0

    # Adding minimum attention value at the end for normalization purposes
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize attention values for each token
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Punctuation and Conjunction Dependency", out