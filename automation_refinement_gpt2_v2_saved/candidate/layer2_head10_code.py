from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple

def detect_function_header(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenizing the input sentence
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Words to consider for function headers:
    function_keywords = {'def', 'import', 'return'}
    function_keyword_indices = []

    # Iterate through the token IDs and words
    for i, token_id in enumerate(toks.input_ids[0]):
        token = tokenizer.decode(token_id)
        # Check if the token is part of function headers
        clean_token = token.strip()
        if clean_token in function_keywords:
            function_keyword_indices.append(i)  # Save index

    # Assign high attention between function header tokens
    for i in function_keyword_indices:
        for j in function_keyword_indices:
            out[i, j] = 1

    # Assign non-zero attention to CLS and EOS tokens (first and last items)
    out[0, 0] = 0.1
    out[-1, -1] = 0.1

    # Normalize by row
    row_sums = out.sum(axis=1, keepdims=True)
    out = np.divide(out, row_sums, where=row_sums != 0)

    return "Function Header Pattern", out