import numpy as np
from transformers import PreTrainedTokenizerBase
import re

def compound_word_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str, np.ndarray:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence using the tokenizer and map subword tokens back to their full forms
    tokens = tokenizer.tokenize(sentence)

    # Identify locations of compound words in tokenized representation
def find_compound_words(tokens):
    compound_indices = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i].startswith('##'):
            start = i - 1
            while i < len(tokens) and tokens[i].startswith('##'):
                i += 1
            compound_indices.append((start, i-1))
        i += 1
    return compound_indices

    # Map token indices
    compound_indices = find_compound_words(tokens)

    # Assume each compound word attended to itself strongly
    for start, end in compound_indices:
        for i in range(start, end+1):
            for j in range(start, end+1):
                out[i+1, j+1] = 1

    # Make sure [CLS] and [SEP] are self-attending
    out[0, 0] = 1
    out[-1, -1] = 1

    # Normalize the matrix
    out = out / out.sum(axis=1, keepdims=True)

    return 'Compound Word Attention Pattern', out