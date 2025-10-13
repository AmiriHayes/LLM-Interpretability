import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

# Define the coreference chain identification function
def coreference_chain(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence for reference
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0], skip_special_tokens=True)

    # We hypothesize that the head focuses on the first mention in a coreference chain and propagates attention throughout the chain
    first_token_index = 1  # Assume the first word is the start of a coreference chain
    first_token = tokens[first_token_index]

    # Loop through each token and distribute attention based on coreferenceural similarity
    for i, token in enumerate(tokens):
        if i != first_token_index:
            if token.lower() in [first_token.lower(), "it", "they", "she", "he", "her", "his"]:
                out[i, first_token_index] = 1  # Attention relies back to the first mention in the coreference

    # Ensure CLS and EOS tokens attend to themselves
    out[0, 0] = 1  # Attention for CLS
    out[-1, -1] = 1  # Attention for EOS

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix row by row to avoid zero division
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Coreference Chain Identification", out