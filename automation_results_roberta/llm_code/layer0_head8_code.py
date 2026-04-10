from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def pronoun_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    pronouns = {'he', 'she', 'it', 'they', 'her', 'his', 'their', 'them'}

    words = sentence.split()
    last_subject_indices = []

    # Map tokens back to words for easier reference
    token_to_word_map = {}
    word_index = 0
    for word, token in zip(words, toks['input_ids'][0]):
        token_to_word_map[token] = word_index
        word_index += 1

    # Identify pronouns and subject-related words to create attention links
    for i, word in enumerate(words):
        if word.lower() in pronouns:
            if last_subject_indices:
                for idx in last_subject_indices:
                    out[i+1, idx+1] = 1  # +1 to account for special [CLS] token
        else:
            # Update last subject indices (e.g., proper nouns, subjects)
            # This part can be more sophisticated
            last_subject_indices.append(i)

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Pronoun Resolution Pattern", out