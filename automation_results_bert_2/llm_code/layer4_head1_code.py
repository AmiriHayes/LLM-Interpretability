import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

# Define a dictionary with tokens and their previous quote 'dependency'
quote_dependent_tokens = {'"': ['before_quote'], '?': ['after_quote'], '.': ['after_quote'],',': ['before_quote']}

# Define the function for 'Quotation Context Attention'
def quotation_context_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the input sentence with necessary mappings for token match
    tokens = tokenizer.tokenize(sentence)
    decoded_tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Align decoded tokens with their respective positions
    token_align_map = {}
    str_index = 0
    for i, dec_tok in enumerate(decoded_tokens):
        if dec_tok in tokens[str_index:]:
            token_align_map[dec_tok] = tokens[str_index:].index(dec_tok) + str_index
            str_index += 1

    # Determine quote context states
    in_quote = False
    last_quote_indices = []
    for i, token in enumerate(decoded_tokens):
        if token == '"':
            in_quote = not in_quote
            last_quote_indices.append(i)
        # Apply dependencies based on quote presence
        if in_quote:
            out[i, last_quote_indices[-1]] = 1  # Attend to last quote
        else:
            dependent_list = quote_dependent_tokens.get(token, [])
            for dep_type in dependent_list:
                if dep_type == 'before_quote' and last_quote_indices:
                    out[i, last_quote_indices[-1]] = 1  # Attend to last quote
                if dep_type == 'after_quote' and len(last_quote_indices)>1:
                    out[i, last_quote_indices[-2]] = 1  # Attend to the quote before last

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Quotation Context Attention", out