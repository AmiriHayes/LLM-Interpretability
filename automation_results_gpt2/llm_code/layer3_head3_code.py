import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def pronoun_coreference_and_entity_linking(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Find all unique entities in the sentence
    words = sentence.split()
    entity_starts = {}
    i = 0
    while i < len(words):
        if words[i].istitle() and words[i] not in ["I", "You", "He", "She", "It", "We", "They"]:
            entity = words[i]
            j = i + 1
            # Include titles like 'Mr.', 'Mrs.', 'Dr.', etc.
            while j < len(words) and words[j].istitle():
                entity += ' ' + words[j]
                j += 1
            entity_starts[i] = entity
            i = j
        else:
            i += 1

    # Attention pattern to reflect coreference and linking to entities
    for start, entity in entity_starts.items():
        for j, word in enumerate(words):
            if word in ["I", "you", "he", "she", "it", "we", "they"] or word in entity.split():
                out[start + 1, j + 1] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if np.sum(out[row]) == 0:
            out[row][-1] = 1.0

    # Normalize attention scores across each row
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Pronoun Coreference and Entity Linking", out