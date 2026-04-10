from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def emphasis_key_nouns_actions(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Using a placeholder method for finding nouns and verbs due to simplicity
    # In a real-world case, you would utilize a tokenizer like spaCy here
    words = sentence.split()
    key_tokens = [i for i, word in enumerate(words) if word.lower() in {"needle", "shirt", "sew", "share", "found", "fix"}]

    # All tokens attend to key nouns and verbs
    for i in range(len_seq):
        for j in key_tokens:
            out[i, j+1] = 1  # +1 because of <s>

    # Ensure no row is all zeros; all tokens can also attend to <s>
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, 0] = 1.0

    # Attention normalization (for demonstrative purposes)
    out += 1e-4  # To prevent division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Emphasis on Key Nouns and Actions", out