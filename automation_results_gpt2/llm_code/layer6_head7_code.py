import numpy as np
from transformers import PreTrainedTokenizerBase

def initial_token_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The hypothesis here is the model's head attends to the initial token or the first special character
    # heavily and other tokens subsequent in the sentence. We'll implement an algorithm that mirrors this.
    # The token that gets maximum attention seems to be the first one in each sentence.
    out[0, :] = 1  # First token attends to all tokens

    for i in range(1, len_seq):
        out[i, 0] = 1  # All other tokens attend heavily to the first token

    # Ensuring no row is entirely zero
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Initial Token Pattern", out