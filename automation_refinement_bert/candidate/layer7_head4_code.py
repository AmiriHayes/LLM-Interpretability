import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

# Load the spacy English model
nlp = spacy.load('en_core_web_sm')


def conjunction_coordination(sentence: str, tokenizer: PreTrainedTokenizerBase) -> np.ndarray:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize and parse with spaCy
    doc = nlp(sentence)

    # Map spaCy tokens to tokenizer tokens
    spacy_to_tokenizer_map = {}
    for token in doc:
        token_start = token.idx
        token_end = token.idx + len(token.text)
        for i, ids in enumerate(toks['input_ids'][0]):
            if toks.token_to_chars(0, i)[0] == token_start:
                spacy_to_tokenizer_map[i] = token.i
                break

    # Identify conjunctions and coordinate elements
    for token in doc:
        if token.pos_ == 'CCONJ':
            # Coordinate conjunction with its adjacent elements
            if token.i - 1 >= 0:
                left_elem = token.i - 1
                right_elem = token.i + 1 if token.i + 1 < len(doc) else token.i
                if spacy_to_tokenizer_map.get(left_elem) is not None:
                    out[spacy_to_tokenizer_map[left_elem], spacy_to_tokenizer_map[token.i]] = 1
                if spacy_to_tokenizer_map.get(right_elem) is not None:
                    out[spacy_to_tokenizer_map[token.i], spacy_to_tokenizer_map[right_elem]] = 1

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Conjunction Coordination Attention", out