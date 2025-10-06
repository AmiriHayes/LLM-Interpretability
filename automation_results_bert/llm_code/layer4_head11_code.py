import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

# Function to model the attention pattern of Layer 4, Head 11

def semantic_role_alignment(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Identifying key semantic pairs and their interactions
    special_tokens = {"[CLS]": 0, "[SEP]": len_seq - 1}
    words = sentence.split()

    # Approach assumes noun/verb interaction is key
    for i, word in enumerate(words):
        if i == 0:  # Skip [CLS]
            continue
        if i >= len(words) - 1:  # Skip [SEP]
            continue
        # Simulate focus on nouns and their related words (verbs/adjectives)
        for j, related_word in enumerate(words):
            if i != j:
                if 'and' in word or 'with' in word:  # e.g., conjunctions
                    out[i+1, j+1] = 0.8
                elif 'her' in word or 'the' in word:  # pronouns and articles
                    out[i+1, j+1] = 0.4

    # Ensure attention doesn't end up being zero
    for row in range(1, len_seq - 1):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Attend to [SEP] to ensure non-zero sum

    # Normalize the matrix row-wise
    out = out / out.sum(axis=1, keepdims=True)

    return "Semantic Role Alignment Pattern", out