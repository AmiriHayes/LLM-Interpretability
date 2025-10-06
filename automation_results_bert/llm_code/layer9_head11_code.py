import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

# Define a function to hypothesize delimiter attention pattern
def delimiter_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Iterate through the tokens
    for i, tok_id in enumerate(toks.input_ids[0]):
        # CLS and SEP get self-attention
        if i == 0 or tok_id == tokenizer.sep_token_id:
            out[i, i] = 1.0
        else:
            # Apply attention between delimiters like commas and periods and the final token
            if i < len_seq - 1 and tokenizer.decode([tok_id]).strip() in {',', '.', '?', '!', ';'}:
                out[i, len_seq - 1] = 1.0  # Let delimiters attend to the SEP token
            elif i == len_seq - 1:
                out[i-1, i] = 1.0  # Allow SEP to attend back to the last real token

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "End-of-Sentence and Delimiter Attention", out