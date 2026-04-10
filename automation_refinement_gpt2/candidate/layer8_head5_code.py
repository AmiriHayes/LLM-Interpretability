import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

# Define function
def interrogative_word_dominance(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize input sentence
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Use tokenizer to map tokens to string
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Identify interrogative words (who, how, what, when, where, why)
    interrogatives = {'who', 'how', 'what', 'when', 'where', 'why'}

    # loop through tokens and mark row and column of each interrogative token
    for i, tok in enumerate(tokens):
        if tok.lower().strip() in interrogatives:
            out[i, :] = 1  # the interrogative dominates across the row pattern
            out[:, i] = 1  # and the column (omni-directional)

    # Ensures the [CLS] and [EOS] tokens are self-attentive
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize
    out = out / out.sum(axis=1, keepdims=True)

    return 'Interrogative Word Dominance', out