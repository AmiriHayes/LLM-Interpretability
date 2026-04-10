import numpy as np
from transformers import PreTrainedTokenizerBase

# Define the main function
def conjunction_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Define potential conjunction tokens
    conjunction_tokens = {"and", "but", "or", "so", ","}

    # Use the tokenizer to identify conjunctions and emphasize them in the attention matrix
    for i, token_id in enumerate(toks.input_ids[0]):
        token_str = tokenizer.decode([token_id]).strip()
        if token_str in conjunction_tokens:
            # Emphasize the conjunction token
            out[i, :] = 1

    # Ensure CLS and SEP tokens are self-attentive to fit the pattern observed in the data
    out[0, 0] = 1  # Emphasize the [CLS] token
    out[-1, 0] = 1  # Emphasize the [SEP] token

    # Normalize the attention matrix
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1  # Ensure no row is all zeros

    out += 1e-4  # Avoid zeros for normalization stability
    out = out / out.sum(axis=1, keepdims=True)  # Normalize each row

    return "Conjunction Emphasis Pattern", out
