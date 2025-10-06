import numpy as np
from transformers import PreTrainedTokenizerBase

# This function models attention for conjunction and sentence coordination patterns
# Predominantly attends to conjunctions, coordinating related clauses

def conjunction_attention_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenizing sentence with a simple split to coordinate with token ids
    words = sentence.split()

    # Mapping from tokenizer to word indices
    # Assuming tokenization aligns mostly with word splits
    word_map = {i: word for i, word in enumerate(words)}

    # Conjunctions typically seen (could also be fetched via sophisticated libraries)
    conjunctions = {"and", "but", "or", "because", ","}

    for i in range(1, len_seq - 1):
        word = word_map.get(i, None)
        if word in conjunctions:
            # Assuming the conjunction coordinates with previous and next word
            if i - 1 > 0:
                out[i - 1, i] = 1  # Previous word
            out[i, i] = 1  # Self-attention
            if i + 1 < len_seq:
                out[i + 1, i] = 1  # Next word

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Default to SEP token (usually last)

    # Normalize the attention matrix
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)
    return "Conjunction Coordination Pattern", out