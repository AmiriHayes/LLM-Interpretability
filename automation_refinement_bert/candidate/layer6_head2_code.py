import numpy as np
from transformers import PreTrainedTokenizerBase

# This function encodes a pattern of focusing on numeric operations.
def numeric_operations_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> np.ndarray:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    tokens = tokenizer.tokenize(sentence)

    number_indices = []
    for idx, token in enumerate(tokens):
        if token.replace('.', '', 1).isdigit() or \
           (token.startswith('##') and token[2:].replace('.', '', 1).isdigit()):
            # If the token is numeric or a continuation of a numeric token
            number_indices.append(idx + 1) # +1 as tokens are offset by [CLS] token

    operation_indices = []
    # Identify operation-like elements like `+`, `-`, `=`
    operation_tokens = set(['+', '-', '=', '##+', '##-', '##='])
    for idx, token in enumerate(tokens):
        if token in operation_tokens:
            operation_indices.append(idx + 1) # +1 as tokens are offset by [CLS] token

    # Assign high weights to interactions between numbers and operations or between numbers themselves
    for num_idx in number_indices:
        for op_idx in operation_indices:
            out[num_idx, op_idx] = 1
            out[op_idx, num_idx] = 1
        for num_idx2 in number_indices:
            if num_idx != num_idx2:
                out[num_idx, num_idx2] = 1

    out += np.eye(len_seq)  # ensure some self-attention
    out[0, 0] = 1  # [CLS]
    out[-1, 0] = 1 # [SEP]

    # Normalize rows of out
    out = out / out.sum(axis=1, keepdims=True)

    return "Focus on Numeric Operations", out