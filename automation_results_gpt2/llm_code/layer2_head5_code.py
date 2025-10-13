import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

# Define the function to predict the conjunction pattern based on linguistic observation.
def conjunction_coordination(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the input sentence.
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assuming SpaCy is used to identify conjunctions (not included in imports since we're using a hypothesized pattern).
    conjunction_indexes = []
    split_sentence = sentence.split()
    for idx, word in enumerate(split_sentence):
        if word in {"and", "or", "but", "so"}:  # Common conjunctions
            conjunction_indexes.append(idx + 1)  # Account for tokenization from GPT2

    # Loop through conjunctions and apply attention to them and the immediate surrounding words (siblings in coordination).
    for idx in conjunction_indexes:
        # Ensure each conjunction attends to its next and previous token, simulating shared context.
        if idx > 0:
            out[idx, idx - 1] = 0.5  # Previous word likely in coordination
        if idx < len_seq - 1:
            out[idx, idx + 1] = 0.5  # Next word likely in coordination
        out[idx, idx] = 1.0  # Self-attention

    # Normalization step to ensure each row sums to 1.0
    out += 1e-4  # Avoid zero rows which could lead to division errors
    out = out / out.sum(axis=1, keepdims=True)

    # Default case to avoid zero rows when no conjunction coordination present (like punctuation).
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Conjunction Coordination Pattern", out