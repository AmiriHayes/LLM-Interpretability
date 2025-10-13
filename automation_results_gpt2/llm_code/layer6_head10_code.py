from transformers import PreTrainedTokenizerBase
from typing import Tuple
import numpy as np

def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Token alignments for key pronouns or names
    key_tokens = [
        "One",  # Acts as a coreference node or a stand-in for following elements
        "She",
        "Lily",
        "Can",
        "Her",
        "Together",
        "It",
        "After",
        "They"
    ]
    words = sentence.split()

    # Create a basic alignment map from tokenizer/token IDs back to words
    word_to_index = {word: i for i, word in enumerate(words)}
    token_indices = [i for i, word in enumerate(words) if word in key_tokens]

    # Simulate attention pattern by making key tokens focus more on their own vicinity and previous key tokens
    for i in token_indices:
        out[i, i] = 1  # Strong self-attention
        if i > 0:
            out[i, i - 1] = 0.5  # Some attention to the previous token for continuity
        if i < len_seq - 1:
            out[i, i + 1] = 0.5  # Some attention to the next token for continuity

    # Normalize to ensure at least some attention output for all tokens
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # If no attention, focus weakly on the EOS token

    out = out / out.sum(axis=1, keepdims=True)  # Normalize by row

    return "Coreference Resolution Pattern", out