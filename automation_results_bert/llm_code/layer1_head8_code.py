import numpy as np
from transformers import PreTrainedTokenizerBase

def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = sentence.split()
    coreference_pairs = {}

    # Example: Manually define pronouns for sentences
    pronouns = {'he', 'she', 'it', 'they', 'them', 'her', 'his', "'s"}
    candidate_referents = set()

    for i, word in enumerate(words):
        if word in pronouns:
            for referent in candidate_referents:
                out[i + 1, referent + 1] = 1
                out[referent + 1, i + 1] = 1
        else:
            candidate_referents.add(i)

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Coreference Resolution Pattern", out