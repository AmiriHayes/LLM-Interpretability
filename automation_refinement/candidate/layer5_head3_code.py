from typing import Tuple
import numpy as np
from transformers import BertTokenizer, PreTrainedTokenizerBase

# This function implements the identified pattern of connecting adjacent commas and conjunctions
# It focuses on predicting the attention pattern where commas and conjunctions often link phrases together.
def connective_conjunction_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Identifying delimiter tokens that usually represent conjunctions or connectors
    delimiter_tokens = {',', 'and', 'but', 'or', 'so', 'yet'}

    # Insert attention mapping for each delimiter token
    for i, token in enumerate(tokens):
        if token in delimiter_tokens:
            # Distribute attention from current token to adjacent words (before and after it)
            if i > 0:
                out[i, i - 1] += 0.5
            if i < len_seq - 1:
                out[i, i + 1] += 0.5

    # Add self-attention for CLS and SEP
    out[0, 0] = 1
    out[-1, -1] = 1

    # Normalize the output matrix
    out = out / out.sum(axis=1, keepdims=True)

    return "Connective Conjunction Attention Pattern", out