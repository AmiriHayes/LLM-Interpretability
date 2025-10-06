from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def pronoun_anaphora_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Split the sentence into words for processing
    words = sentence.split()
    index_toks = {(i, toks.word_ids(0)[i]): i for i in range(len_seq)
            if toks.word_ids(0)[i] is not None}

    pronouns = {"he", "she", "it", "they", "her", "his", "their", "them", "him"}

    for i in range(len(words) - 1):
        word = words[i]
        word_id = [k for k, v in index_toks.items() if v == i]
        if not word_id:
            continue

        if word.lower() in pronouns:
            for j in range(i - 1, -1, -1):
                previous_word = words[j]
                previous_word_id = [k for k, v in index_toks.items() if v == j]
                if not previous_word_id:
                    continue

                if previous_word.lower() not in pronouns:
                    out[word_id[0], previous_word_id[0]] = 1
                    break

    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Pronoun-Anaphora Resolution", out