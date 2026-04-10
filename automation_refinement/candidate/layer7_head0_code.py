import numpy as np
from transformers import PreTrainedTokenizerBase

def second_word_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign weights to the second token in the sentence
    # Initialize with all attentions to second token getting high values
    for i in range(1, len_seq - 1):
        out[i, 1] = 1

    # Handle special tokens separately
    out[0, 0] = 1   # CLS token attending to itself
    out[-1, -1] = 1 # SEP token attending to itself

    # Normalize attention scores
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)
    return "Second Word Attention Pattern", out