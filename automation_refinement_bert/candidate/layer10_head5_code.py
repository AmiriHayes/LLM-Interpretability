import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

# Hypothesis: This head focuses on numerical components in sentences.
def numerical_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign higher weights to tokens surrounding numeric values
    for i, token_id in enumerate(toks.input_ids[0]):
        token_str = tokenizer.decode([token_id]).strip()
        if token_str.replace('.', '', 1).isdigit() or token_str in {',', '$', '%'}:
            if i > 0:
                out[i, i-1] = 1  # Focus on preceding token
            out[i, i] = 1.5  # Stronger self-focus
            if i < len_seq - 1:
                out[i, i+1] = 1  # Focus on following token
        elif token_str == '.' or token_str == ',':
            out[i, i] = 0.5  # Lower self-focus for punctuation

    out[0, 0] = 1  # Self-focus for [CLS]
    out[-1, 0] = 1  # Focus on [SEP]

    # Normalize the output matrix
    out += 1e-4  # Small constant to ensure numerical stability
    out = out / out.sum(axis=1, keepdims=True)

    return 'Numerical Focus Pattern', out
