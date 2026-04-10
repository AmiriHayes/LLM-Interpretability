import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def contextual_modifier_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence using the tokenizer
    tokenized_sentence = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    emphasis_tokens = {"of", "in", "with", "to"}  # Common phrases that indicate emphasis or connection

    # Assign attention to contextual modifiers
    for i, token in enumerate(tokenized_sentence):
        if token in emphasis_tokens:
            try:
                # Apply attention to surrounding context, up to one token left and right
                if i > 1:
                    out[i, i-1] = 1
                if i < len_seq - 1:
                    out[i, i+1] = 1
            except IndexError:
                pass
        else:
            out[i, i] = 1  # Fallback: self attention

    # Ensure [CLS] and [SEP] tokens have some self attention
    out[0, 0] = 1
    out[-1, -1] = 1

    # Normalize each row
    out += 1e-4  # Add a small epsilon for numerical stability
    out = out / out.sum(axis=1, keepdims=True)

    return "Contextual Modifier Emphasis", out