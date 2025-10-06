from transformers import PreTrainedTokenizerBase
import numpy as np


def punctuation_and_sentence_delimiters(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Define punctuation tokens' attention pattern
    punctuation_attention = {",", ".", "?", "!", "[SEP]", "[CLS]"}

    # Go through tokens and assign stronger attention to punctuation/delimiters
    for i in range(1, len_seq - 1):
        token = tokenizer.decode(toks.input_ids[0][i].item())
        # Remove spaces when comparing tokens
        token = token.strip()
        if token in punctuation_attention:
            out[i, :] = 1 / len_seq
        else:
            out[i, i] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the matrix rows to sum to 1
    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Punctuation and Sentence Delimiters", out