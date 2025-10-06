from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def instrument_tool_relationship(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Define a basic mapping for tools and instruments
    tool_keywords = {'needle', 'button', 'shirt'}
    instrument_keywords = {'sew', 'fix', 'share'}

    # Tokenize and label tokens
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # For each token, check if it matches any tool or instrument keyword
    for i, token in enumerate(tokens):
        if token in tool_keywords:
            tool_index = i
            # They often are paired with preceding or following instruments
            for j, candidate in enumerate(tokens):
                if candidate in instrument_keywords:
                    out[j, tool_index] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize output matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Instrument-Tool Relationship", out