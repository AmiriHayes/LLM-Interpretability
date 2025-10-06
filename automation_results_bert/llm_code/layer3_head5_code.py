import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def conjunction_parsing(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Use special tokens to always connect [CLS] and [SEP] to themselves
    out[0, 0] = 1  # [CLS] attends to itself
    out[-1, -1] = 1  # [SEP] attends to itself

    # Identifying coordination patterns
    for i, token_id in enumerate(toks.input_ids[0]):
        # Assuming tokenizer provides the tokens in a clean manner
        token_str = tokenizer.convert_ids_to_tokens([token_id])[0]

        # Highlight simple conjunctions 'and', 'but', 'because'
        if token_str in ['and', 'but', 'because']:
            # Usually, the word coordinates with the word immediately following it & previous word
            if i > 0:  # Ensure there is a prior token
                out[i, i-1] = 1
            if i < len_seq - 1:  # Ensure there is a subsequent token
                out[i, i+1] = 1

        # Ensure conjunctions also connect reciprocally
        for j in range(len_seq):
            if out[i, j] == 1:
                out[j, i] = 1

    # Ensure normalization by row to simulate model pattern behavior
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Conjunction Parsing and Coordination", out