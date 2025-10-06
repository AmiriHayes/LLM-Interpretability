import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple
import re

# Hypothesis: This head pays special attention to conjunctions and connectives in sentences,
# emphasizing the relationships between clauses or phrases they connect.


CONJUNCTIONS = {'and', 'but', 'or', 'yet', 'for', 'nor', 'so', 'because', 'if', 'when', 'as', 'while', 'although', 'even though', 'through', 'since', 'though'}

# Function to check if a word is a conjunction or connective

def is_conjunction(word):
    return word.lower() in CONJUNCTIONS


def conjunction_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence
    tokenized_sentence = sentence.split()

    # Identify conjunctions and set attention
    for i, token in enumerate(tokenized_sentence):
        if is_conjunction(token):
            for j in range(1, len_seq - 1):  # Ignore CLS [0] and SEP [len_seq-1]
                if j != i:  # Do not emphasize self
                    out[i + 1, j] = 1  # Token starts at 1 due to potential [CLS]

    # Normalize attention by row and adding self- and SEP-attention
    for row in range(len_seq):
        if out[row].sum() == 0:  # Ensure no row is all zeros
            out[row, -1] = 1.0  # SEP attention

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Conjunction and Connectives Attention Pattern", out