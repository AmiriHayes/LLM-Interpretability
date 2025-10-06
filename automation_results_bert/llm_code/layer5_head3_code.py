import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def conjunction_and_coordination(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Define a simple set of conjunctions for this example
    conjunctions = {"and", "but", "or", "so", "because"}

    # Use the tokenizer to align token indices
    words = [tokenizer.decode(tok).strip() for tok in toks.input_ids[0]]  # Decode tokens into words
    word_to_index = {word: idx for idx, word in enumerate(words)}  # Map words to their indices

    # Iterate over words and apply attention based on the presence of conjunctions
    for i, word in enumerate(words):
        if word in conjunctions:
            # A conjunction typically relates previous and following tokens
            for j in range(i):  # Look backward
                out[i, j] = 1
            for j in range(i + 1, len_seq):  # Look forward
                out[i, j] = 1

    # Ensure CLS and SEP have attention to avoid any row being all zeros
    out[0, :] = 1  # Self-attention for [CLS]
    out[:, 0] = 1  # Attend to [CLS]
    out[-1, :] = 1  # Self-attention for [SEP]
    out[:, -1] = 1  # Attend to [SEP]

    # Normalize attention matrix row-wise
    out += 1e-4  # Small value to prevent division errors
    out = out / out.sum(axis=1, keepdims=True)

    return "Conjunctions and Coordination", out