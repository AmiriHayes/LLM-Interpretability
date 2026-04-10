import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def operator_numerical_constants(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence into tokens
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    operators = {'+', '-', '*', '/', '=', '^'}
    numerical_constants = set('0123456789')

    # Loop through the tokens and identify operators and numerical constants
    for i, token in enumerate(tokens):
        # Check if the token is an operator or a numerical constant
        if any(char in operators for char in token) or any(char in numerical_constants for char in token):
            # Heuristic to attend to nearby tokens if a token is an operator or numeral constant
            for j in range(max(0, i - 1), min(len_seq, i + 2)):
                out[i, j] = 1.0

    # Normalize the output matrix
    out = out / out.sum(axis=1, keepdims=True)

    return "Operator and Numerical Constants Identification", out