from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

# Function to identify the sentence's primary element or backbone, mainly focusing attention on the leading words.
def sentence_backbone(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Attention pattern focuses heavily on the first non-punctuation, non-article token / verb or noun.
    # This approach targets the element that is often acting as a central theme or structure in syntactic terms.
    words = sentence.split()
    for i, word in enumerate(words):
        # Tokenize by spaCy to ensure consistent alignment.
        if not word.lower() in {"the", "a", "an", ",", ".", "!", "?", "'", ":"}:
            for j in range(1, len_seq-1):
                out[j, i+1] = 1
            break

    # Adding default attention to first and last special tokens
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalizing by rows to simulate a realistic attention distribution
    out += 1e-4  # Small value to account for zero sum
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Backbone Identification", out