from transformers import PreTrainedTokenizerBase
import numpy as np

# Function implementation

def subject_association(sentence: str, tokenizer: PreTrainedTokenizerBase):
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Use a simple rule: First token is assumed to be the subject (e.g., first noun or pronoun)
    subject_index = 0
    out[subject_index, subject_index] = 1

    # Each token will attend to the first token, which is assumed to represent the subject
    for i in range(1, len_seq-1):
        out[i, subject_index] = 1

    # Ensure the CLS (beginning) and EOS (end) token have some attention set for decoding purposes
    out[0, 0] = 1
    out[-1, -1] = 1

    # Normalize the rows of the attention matrix to simulate probability distributions
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # To prevent division by zero
    out = out / row_sums

    return "Subject Association Pattern", out