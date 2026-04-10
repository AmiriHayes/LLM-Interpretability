import numpy as np
from typing import Tuple

def pronoun_coreference(sentence: str, tokenizer) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    pronouns = ["he", "she", "him", "her", "they", "them", "it", "its", "his", "hers"]

    for i, tok in enumerate(words):
        if tok in pronouns:
            # Simple heuristic: link to the closest preceding noun
            for j in range(i - 1, 0, -1): # Iterate backwards, excluding [CLS] and current
                # Check for capitalized words as a proxy for nouns (simplification)
                if words[j].isalpha() and words[j][0].isupper():
                  out[i, j] = 0.7  # Attend to the noun
                  out[i, i] = 0.3 # Attend to self
                  break  # Link to closest preceding noun

    out[0, 0] = 1 # cls attends to self
    out[-1, 0] = 1 # sep attends to cls
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize rows

    return "Pronoun Coreference Pattern", out