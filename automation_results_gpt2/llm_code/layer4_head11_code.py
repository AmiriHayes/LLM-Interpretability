import numpy as np
from transformers import PreTrainedTokenizerBase

def co_reference_and_coordination(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    conjunctions = {"and", "but", "or", "so"}

    for i, word in enumerate(words):
        # Self-attention for nouns and pronouns in co-reference resolution
        if "Lily" in word or "mom" in word or "girl" in word or "she" in word or "it" in word:
            out[i, i] = 1.0

        # Coordination attention logic
        if word in conjunctions:
            # Look back for nouns or pronouns
            for j in range(i - 1, 0, -1):
                if "Lily" in words[j] or "mom" in words[j] or "girl" in words[j] or "it" in words[j] or "she" in words[j]:
                    out[j, i] = 1.0
                    break
            # Look forward for a second noun or verb
            for j in range(i + 1, len_seq):
                if "Lily" in words[j] or "mom" in words[j] or "girl" in words[j] or "it" in words[j] or "she" in words[j]:
                    out[j, i] = 1.0
                    break

    # Ensure no row is completely zero
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Some attention to the final token

    return "Co-referential and Coordination Attention", out