import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')  # Load spaCy English model


def clause_oriented_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize sentence with spaCy for possible sentence split tokens
    doc = nlp(sentence)

    # Dictionary to keep track of subword alignment between tokenizer and spaCy
    alignment = {}
    token_idx = 0

    for token in doc:
        # Get span of each token in the spaCy doc
        span = token.idx, token.idx + len(token)
        subword_ids = tokenizer.encode_plus(sentence[span[0]:span[1]], add_special_tokens=False)["input_ids"]

        # Map each subword to its corresponding word
        for sub_id in subword_ids:
            alignment[token_idx] = sub_id
            token_idx += 1

    # We will treat tokens within the same clause as having strong attention
    clause_ends = {';', ',', '.', '"', '?', '!'}
    last_segment_end = 0

    for i, token in enumerate(toks.input_ids[0]):
        if "</s>" in tokenizer.convert_ids_to_tokens(token.item()):
            last_segment_end = i
            break
        token_str = tokenizer.convert_ids_to_tokens(token.item())
        if token_str in clause_ends:
            for head in range(last_segment_end, i + 1):
                for dep in range(last_segment_end, i + 1):
                    out[head, dep] = 1  # Assign attention within the clause
            last_segment_end = i + 1

    # Ensure no row is empty
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize row-wise

    return "Clause-Oriented Attention Pattern", out