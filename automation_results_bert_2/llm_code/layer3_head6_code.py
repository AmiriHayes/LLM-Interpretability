import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

# Define the function to interpret attention patterns

def conjunction_and_quotation(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    # Initialize the output attention matrix
    out = np.zeros((len_seq, len_seq))
    # Define special tokens and conjunctions
    conjunctions = {"and", "because"}
    quote_tokens = {'"', '"'}

    # Convert tokens to text
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Populate the attention matrix based on conjunctions
    for i, tok in enumerate(tokens):
        if tok in conjunctions:
            out[i, i] = 1  # self-attention
            if i + 1 < len_seq:
                out[i, i + 1] = 1  # attend to the following token
        if tok in quote_tokens:
            out[i, i] = 1  # self-attend to opening and closing quotes
            # Optionally attend to next token considering a context condition
            if i + 1 < len_seq:
                out[i, i + 1] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize rows

    return "Conjunction and Quotation Marks Pattern", out