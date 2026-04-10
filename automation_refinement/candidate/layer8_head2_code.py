import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def list_and_conjunction_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Parse the sentence using spaces (potentially for further analysis if needed)
    words = sentence.split()

    # Track lists and conjunctions
    for i in range(1, len_seq-1):
        token = toks.tokens()[i]
        # Mark typical conjunctions or list indicating tokens
        if token in [',', 'and', 'or']:  
            # Attend to other list/conjunction tokens
            for j in range(1, len_seq-1):
                other_token = toks.tokens()[j]
                if other_token in [',', 'and', 'or']:
                    out[i, j] = 1  # Strong attention to other similar tokens

    # Ensure attention to CLS and SEP tokens
    out[0, 0] = 1
    out[-1, 0] = 1  

    # Normalize the attention by row to simulate uniform distribution
    out += 1e-4  # Small value to ensure numerical stability
    out = out / out.sum(axis=1, keepdims=True)

    return "List Item and Conjunction Association", out