from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def object_to_descriptor_linkage(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Rule for linking objects with descriptor or adjective-like parts
    words = sentence.split()
    descriptors = ['hue', 'problem', 'sky', 'day', 'sense', 'secrets', 'story', 'adventure', 
                   'journey', 'kitchen', 'symphony', 'language', 'routine', 'idea']

    # Simple linkage pattern for object-descriptor relations
    for i, word in enumerate(words):
        if word in descriptors:
            matching_indices = [idx for idx, part in enumerate(words) if part in word]
            for idx in matching_indices:
                out[i+1, idx+1] = 1

    out[0, 0] = 1  # CLS token attends to itself
    out[-1, 0] = 1 # SEP token attends to CLS

    # Normalize attention matrix
    out += 1e-4
    out /= out.sum(axis=1, keepdims=True)

    return 'Object to Descriptor Linkage', out