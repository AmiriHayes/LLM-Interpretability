import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def function_definition_identifier(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = sentence.split()
    func_indices = [i for i, word in enumerate(words) if word == 'def']

    token_word_map = {}
    current_token_index = 0
    for word in words:
        num_tokens = len(tokenizer.tokenize(word))
        for _ in range(num_tokens):
            token_word_map[current_token_index] = word
            current_token_index += 1

    for idx in func_indices:
        start_token = list(token_word_map.keys())[list(token_word_map.values()).index(words[idx])]
        out[start_token, start_token] = 1
        for j in range(start_token + 1, len_seq):
            if token_word_map[j] == '(':  # end token for function name
                break
            out[j, start_token] = 1

    # Self-attention for [CLS] and [SEP] equivalents
    out[0, 0] = 1  # Usually [CLS] or starting token
    out[-1, 0] = 1  # Mapping to [CLS] or starting context

    # Normalize the attention weights by row
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Function Definition Identifier", out