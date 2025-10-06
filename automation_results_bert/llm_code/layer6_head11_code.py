import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')


def conjunction_coordination(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence with spacy to get linguistic features
    doc = nlp(sentence)
    token_map = {tok.i: i for i, tok in enumerate(doc)}  # maps token index in spaCy to Bert/Transformers index

    for tok in doc:
        if tok.dep_ in {"cc", "conj"} or tok.text in {"and", "but", "or"}:
            head_index = tok.head.i
            for child in tok.head.children:
                if child.dep_ in {"conj", "cc"} or child.text in {"and", "but", "or"}:
                    current_index = token_map.get(tok.i, -1)
                    sibling_index = token_map.get(child.i, -1)
                    head_index_token = token_map.get(head_index, -1)

                    if current_index >= 0 and sibling_index >= 0:
                        out[current_index, sibling_index] = 1
                        out[sibling_index, current_index] = 1
                    if head_index_token >= 0:
                        out[head_index_token, sibling_index] = 1
                        out[sibling_index, head_index_token] = 1

    # Ensure no row is all zeros by attending to [SEP] token
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Conjunction Coordination", out