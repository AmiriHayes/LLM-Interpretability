import numpy as np
from transformers import PreTrainedTokenizerBase


def parse_math_expression(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = sentence.split()
    # Math symbols of interest
    math_symbols = {'+', '-', '=', '*', '/', '(', ')', '<', '>', '\', '^'}

    # Loop over tokens and align attention
    for i, word in enumerate(words):
        if any(sym in word for sym in math_symbols):
            base_token_index = i + 1  # Index in the tokenized input
            for j, comp_word in enumerate(words):
                if i != j and any(sym in comp_word for sym in math_symbols):
                    comp_token_index = j + 1
                    out[base_token_index, comp_token_index] = 1
                    out[comp_token_index, base_token_index] = 1

    # Assign special CLS and EOS token attention for overall sentence context
    out[0, 0] = 1  # CLS token attends to itself
    out[-1, 0] = 1  # EOS token slightly attends to the sentence start

    # Normalize the attention
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return 'Mathematical Expression Parsing Pattern', out