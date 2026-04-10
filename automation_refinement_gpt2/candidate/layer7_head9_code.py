from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def initial_dominance_by_first_token(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # The pattern of attention is dominated by attention from the first token to all others.
    # Identify the sentence initial token (ignoring leading white spaces if any)
    leading_non_space = next((i for i, tok_id in enumerate(toks.input_ids[0]) if tok_id != tokenizer.pad_token_id), None)
    if leading_non_space is not None:
        # Assign maximum attention from the sentence initial token to others 
        out[leading_non_space, :] = 1.0

    # Normalize attention to match typical softmax pattern
    out = out / np.sum(out, axis=1, keepdims=True)

    return 'Sentence Initial Dominance by First Token', out