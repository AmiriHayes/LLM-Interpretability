from transformers import PreTrainedTokenizerBase
from typing import Tuple
import numpy as np

# Function to predict focus on subject and object importance

def subject_object_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Split the sentence into words, and assume alignment with tokenizer as each token corresponds to a word (simplistic)
    words = sentence.split()

    # Set key attention features manually from hypothesized behavior
    # Primary focus is top-down along sequence, connections to main verbs
    # Let's hypothesize verbs are generally the 3rd or 4th word (indicative placeholder)

    for i, word in enumerate(words):
        if i > 0 and word.lower() in ["was", "knew", "said", "went", "found", "shared", "because", "could"]:
            # Simulate focus from subjects and objects towards these verbs
            for j in range(len_seq):
                if j != i and toks.input_ids[0][j] != 101 and toks.input_ids[0][j] != 102:  # ignore CLS and SEP
                    out[j, i] = 1

    # Always give special tokens some reference if their row ends up having no attention
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix by row dividing out row sums
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Focus on Subject and Object Connections", out