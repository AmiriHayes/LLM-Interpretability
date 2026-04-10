import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

# The hypothesis is that Layer 4, Head 1 attends strongly to the initial subject or initial part of the sentence across most tokens.
def initial_subject_continuation(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # We will use spacy for part-of-speech tagging to identify the initial noun phrase
    import spacy
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])
    doc = nlp(sentence)

    # Find the span of the initial noun phrase or subject-related phrase, until the first verb or significant punctuation
    noun_end = 0
    for token in doc:
        if token.pos_ in {'VERB', 'PUNCT'}:
            break
        noun_end = token.i + 1

    # Assuming token mapping between spacy and tokenizer is straightforward for initial span
    for i in range(1, len_seq - 1):
        for j in range(1, noun_end + 1):
            out[i, j] = 1

    # Self-attention with initial subject influence
    out[0, 0] = 1  # CLS token attention
    out[-1, 0] = 1  # EOS token attention

    # Normalize attention
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return 'Initial Subject Continuation', out
