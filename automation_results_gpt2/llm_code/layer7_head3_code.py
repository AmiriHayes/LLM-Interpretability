import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

# This function hypothesizes that Layer 7, Head 3 in GPT-2 attends strongly to a root word or primary subject,
# namely the initial word of the sentence or a closely related noun, particularly in narrative text.

def sentence_root_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence and predict attention pattern
    tokens = tokenizer.convert_ids_to_tokens(toks['input_ids'][0])

    # Assume that the first token is the sentence root, if it isn't special (CLS, SEP)
    root_token_index = 0

    # Assign high attention to the root token from other tokens
    for i in range(1, len_seq - 1):
        out[i][root_token_index] = 1.0

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize to simulate what a model would presumably have in terms of probabilities
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Root Attention Pattern", out