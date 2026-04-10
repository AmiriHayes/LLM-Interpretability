from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def comma_phrase_association(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = sentence.split()
    doc = nlp(sentence)

    # Token IDs from tokenizer may differ from spaCy's. Create alignment
    token_dict = {toks.token_to_chars(i)[0]: i for i in range(len_seq)}

    # Identify phrase and their comma-based separators
    for tok_index, token in enumerate(doc):
        if token.text == ",":
            # Left and right context
            left = [child for child in token.lefts]
            right = [child for child in token.rights]
            # Apply associations within context
            for l_token in left:
                for r_token in right:
                    out[token_dict.get(l_token.idx, 0), token_dict.get(r_token.idx, len_seq - 1)] = 1
                    out[token_dict.get(r_token.idx, 0), token_dict.get(l_token.idx, len_seq - 1)] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize matrix

    return "Comma-Phrase Association", out