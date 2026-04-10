from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple


def subject_verb_alignment(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = sentence.split()
    for i, word in enumerate(words[:-1]):
        # Assuming simple rule for subject-verb
        if word.lower() in ["he", "she", "it", "they", "we", "you", "I"]:
            for j in range(i + 1, len(words)):
                if words[j].lower() in ["is", "are", "was", "were", "have", "has", "do", "does", "did"]:
                    # Set attention from subject (i) to following verb (j)
                    out[i + 1, j + 1] = 1
                    break

    # Ensure CLS has some attention and SEP is considered
    out[0, -1] = 1
    out[-1, 0] = 1

    for row in range(len_seq):
        # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Subject-Verb Alignment Pattern", out