from typing import Tuple
from transformers import PreTrainedTokenizerBase
import numpy as np

# Hypothesis: The head focuses on the subject of the sentence, often the pronouns or the main entities, emphasizing the first noun or pronoun in the sentence.

def pronoun_subject_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    first_noun_index = -1
    start_special_token = 0
    # Find the first non-special token which is usually the first pronoun or subject.
    for idx, tok_id in enumerate(toks['input_ids'][0]):
        if tokenizer.decode(tok_id.item()).strip() not in tokenizer.all_special_tokens:
            first_noun_index = idx
            break

    # If a valid first noun/pronoun is found, assign strong attention from and to this token.
    if first_noun_index != -1:
        out[first_noun_index, first_noun_index] = 1.0
        for j in range(len_seq):
            if j != first_noun_index:
                out[first_noun_index, j] = 0.5
                out[j, first_noun_index] = 0.5

    # Ensure no row is all zeros (safeguard against division by zero errors in subsequent normalization).
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Smooth the matrix to prevent division by zero.
    out = out / out.sum(axis=1, keepdims=True)  # Normalize across rows to simulate attention.

    return "Pronoun Reference and Sentence Subject Emphasis", out