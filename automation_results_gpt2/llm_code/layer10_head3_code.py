from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def high_semantic_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Identify content-heavy words in the beginning of the sentence
    # Assume content words are typically nouns, verbs, etc. from the beginning
    # We are focusing mostly on first few tokens and spreading attention from them
    # Define a simplistic heuristic for this test case
    primary_focus = min(len_seq - 2, 5)  # Focus on a max of 5 tokens initially

    # Assigning attention weight to tokens towards the start
    # Set the '[CLS]' token to attend to key words at the beginning
    for i in range(1, primary_focus):
        out[0, i] = 1  # Increase attention from CLS to these words
        for j in range(1, len_seq-1):
            out[i, j] = 1

    # Normalize the rows
    out = out + 1e-6  # Prevent division by zero
    out /= out.sum(axis=1, keepdims=True)

    # Return the pattern type
    return "High Semantic Value Start Pattern", out