import numpy as np
from transformers import PreTrainedTokenizerBase
import re
from collections import defaultdict

# Define the function to calculate predicted attention

def repetitive_entity_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize words by space for pattern matching
    words = sentence.split()

    # Use regex patterns for simple token matching
    word_positions = defaultdict(list)
    pattern = re.compile(r'[a-zA-Z0-9]+')
    for idx, word in enumerate(words):
        match = pattern.match(word)
        if match:
            # Use lower case for case insensitivity
            word_positions[match.group(0).lower()].append(idx + 1)

    # Assign attention based on repetitive entities
    for positions in word_positions.values():
        if len(positions) > 1: # Entity must be repetitive
            for pos in positions:
                for other_pos in positions:
                    if pos != other_pos: # Avoid self-attention
                        out[pos, other_pos] = 1

    # Ensure [CLS] and [SEP] have some attention
    out[0, 1:] = 1 / (len_seq - 1)
    out[-1, :-1] = 1 / (len_seq - 1)

    # Ensure all rows are normalized
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
        else:
            out[row] += 1e-4  # Avoid division by zero
            out[row] = out[row] / out[row].sum()  # Normalize

    return "Repetitive Entity Attention", out