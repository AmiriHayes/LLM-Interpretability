import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

# Define Pronoun-Referent Relationship pattern function
def pronoun_referent_relationship(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    tokens = toks.input_ids[0].tolist()
    token_strs = [tokenizer.convert_ids_to_tokens(tok) for tok in tokens]

    # Simple pronoun list
    pronouns = {'her', 'his', 'its', 'their', 'they', 'he', 'she', 'it', 'you', 'we', 'our', 'its'}

    # Iterate through tokens to find pronoun-referent pairs
    for i in range(1, len_seq - 1):
        for j in range(i + 1, len_seq - 1):
            if token_strs[i] in pronouns and token_strs[j] not in pronouns:
                # We assume a strong attention from pronouns to their likely referents
                out[i, j] = 1
            elif token_strs[j] in pronouns and token_strs[i] not in pronouns:
                # We also check the opposite direction
                out[j, i] = 1

    # Ensure rows are not all zero
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize rows to sum to 1
    out = out / out.sum(axis=1, keepdims=True)

    return "Pronoun-Referent Relationship", out