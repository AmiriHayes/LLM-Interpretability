import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple


def function_definition_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    word_to_token = toks.word_ids

    # Variables to catch specific tokens
    identifier_indices = []
    definition_indices = []

    # Using a dictionary to map tokenizers from transformers and spaCy if needed
    def is_function_def_token(token_idx):
        # naive condition to identify function tokens like 'def', or identifiers and their blocks
        token_text = tokenizer.decode(toks.input_ids[0][token_idx])
        if 'def' in token_text or token_text.strip():
            return True
        return False

    for i in range(len_seq):
        if word_to_token[i] is not None:
            if is_function_def_token(i):
                definition_indices.append(i)

            # Track indices of function identifiers/definitions
            if 'def' in tokenizer.decode(toks.input_ids[0][i]):
                identifier_indices.append(i)

    # For the function definitions, have them self-attend strongly and reference each other
    for definition_index in definition_indices:
        out[definition_index, definition_index] = 1
        for identifier_index in identifier_indices:
            if definition_index != identifier_index:
                out[definition_index, identifier_index] = 1

    # Add some baseline attention for CLS and SEP
    out[0, 0] = 1  # CLS token or start
    out[len_seq - 1, 0] = 1  # SEP token or end

    # Normalize rows to sum up to 1
    out = out / out.sum(axis=1, keepdims=True)

    return "Function Definition and Identifier", out