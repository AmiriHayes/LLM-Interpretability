import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def attention_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # High attention to <s> and </s> tokens
    out[:, 0] = 0.5  # Allocate significant attention to <s>
    out[:, -1] = 0.5 # Allocate significant attention to </s>

    # Emphasize connective words
    connective_words_indices = []
    word_ids = toks.word_ids()
    words = "".join(sentence.split()).split('\u0120')

    # Define a simple list of connective words
    connectives = {'and', 'because', 'but', 'so'}

    # Assign higher attention values to connective words
    for i, word in enumerate(words):
        if any(conn in word for conn in connectives):
            if i < len(word_ids) and word_ids[i] is not None:
                connective_words_indices.append(word_ids[i])

    for index in connective_words_indices:
        out[index, 0] = 0.25  # Attach less than boundary tokens to connectives
        out[index, -1] = 0.25

    # Normalize the matrix so that each row sums to 1
    row_sums = out.sum(axis=1, keepdims=True)
    out = np.divide(out, row_sums, where=row_sums!=0)  # Avoid division by zero

    return "High Attention to Sentence Boundaries and Connective Words", out