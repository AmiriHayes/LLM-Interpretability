import numpy as np
from transformers import PreTrainedTokenizerBase

def function_definition_header_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    # Tokenize the sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Prepare to decode back into common tokens (GPT-2 and spacey tokenization may differ)
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    token_indices = {idx: token for idx, token in enumerate(tokens)}

    # Rule 1: Head attends to 'def' indicating a function definition
    def_indices = [idx for idx, token in token_indices.items() if 'def' in token]

    # Assign attention to indices close to a 'def'
    for def_idx in def_indices:
        if def_idx < len_seq:
            # Give high attention to tokens directly after the function 'definiton'
            out[def_idx, def_idx] += 1
            out[def_idx, min(def_idx + 1, len_seq - 1)] = 1

    # Rule 2: Default pattern where each token gives some self-attention
    for i in range(len_seq):
        out[i, i] = 0.1

    # Assign special treatment for CLS and EOS tokens
    out[0, 0] = 1  # CLS attention
    out[-1, 0] = 1  # EOS -> CLS

    # Normalize to simulate probability distribution
    out /= out.sum(axis=1, keepdims=True)

    return "Function Definition Header Focus", out