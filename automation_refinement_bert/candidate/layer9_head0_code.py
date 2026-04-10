from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

# This function captures the hypothesized attention pattern
# where tokens that are semantically related, like synonyms
# or words belonging to similar conceptual categories, are attended to each other

def semantic_field_association(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    doc = nlp(sentence)
    semantically_similar_dict = {word: token.i for token in doc for word in doc if (word != token and word.tag_ == token.tag_ and word.vector.norm() > 0 and token.vector.norm() > 0 and word.similarity(token) > 0.5)}
    for key_token, key_index in semantically_similar_dict.items():
        for similar_word in semantically_similar_dict:
            if similar_word != key_token:
                similar_index = semantically_similar_dict[similar_word]
                out[key_index + 1, similar_index + 1] = 1
                out[similar_index + 1, key_index + 1] = 1

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Semantic Field Association", out


