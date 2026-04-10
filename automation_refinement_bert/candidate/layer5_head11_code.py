import numpy as np
from transformers import PreTrainedTokenizerBase

def consistent_dependency_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> ('Punctuation Consistent Dependency Attention', np.ndarray):
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # To identify punctuation positions for consistent attention 
    punctuation_tokens = {',', '.', '?', '!', ':', ';', '-', '(' , ')', '[', ']', '{', '}', '"', "'"}

    # Decode token ids to tokens
    decoded_tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    token_to_index = {index: tok for index, tok in enumerate(decoded_tokens, 0)}

    # Find indexes of punctuation tokens
    punctuation_indexes = [i for i, token in token_to_index.items() if token in punctuation_tokens]

    # Create consistent dependency: all punctuation token positions will attend to each other uniformly
    for idx in punctuation_indexes:
        for j in punctuation_indexes:
            if idx != j:
                out[idx, j] = 1 / (len(punctuation_indexes) - 1)

    # Filling other interactions with weighted attention to next token after punctuation
    for idx, token in token_to_index.items():
        if idx in punctuation_indexes and idx + 1 < len_seq:
            out[idx + 1, idx] = 1.0

    # Normalization: Ensure row sums to 1
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
        else:
            out[row] = out[row] / out[row].sum()

    return "Punctuation Consistent Dependency Attention", out