import numpy as np
from typing import Tuple
from transformers import AutoTokenizer

def punctuation_next_word_alignment(sentence: str, tokenizer: AutoTokenizer) -> Tuple[str, np.ndarray]:
    """
    Hypothesizes the attention pattern for Layer 6, Head 6, which links punctuation
    marks to the token that immediately follows them. This pattern is crucial
    for bridging structural breaks and list items in a sentence.

    Args:
        sentence (str): The input sentence string.
        tokenizer: The tokenizer to use (e.g., AutoTokenizer).

    Returns:
        Tuple[str, np.ndarray]: A tuple containing the pattern name and the
                                predicted attention matrix of size (token_len, token_len).
    """
    encoded_input = tokenizer(
        sentence,
        return_tensors="pt",
        add_special_tokens=True,
        return_offsets_mapping=True
    )
    input_ids = encoded_input.input_ids[0]
    word_ids = encoded_input.word_ids(batch_index=0)
    token_len = len(input_ids)
    
    predicted_matrix = np.zeros((token_len, token_len))

    # Identify special punctuation tokens
    punctuation_ids = {
        tokenizer.convert_tokens_to_ids(','),
        tokenizer.convert_tokens_to_ids('.'),
        tokenizer.convert_tokens_to_ids('?'),
        tokenizer.convert_tokens_to_ids('!'),
        tokenizer.convert_tokens_to_ids(':'),
        tokenizer.convert_tokens_to_ids(';'),
        tokenizer.convert_tokens_to_ids("'"),
        tokenizer.convert_tokens_to_ids('"'),
        tokenizer.convert_tokens_to_ids('-')
    }

    # Loop through all tokens to find punctuation and create attention link
    for i in range(token_len):
        token_id = input_ids[i].item()
        
        # Check if the current token is a punctuation mark
        if token_id in punctuation_ids:
            # Check for the next token, ensuring it's not a special token
            if i + 1 < token_len and word_ids[i+1] is not None:
                predicted_matrix[i, i+1] = 1.0

    # Handle multi-token words.
    # The head also shows a pattern of high self-attention for sub-tokens.
    # We will encode this by making all sub-tokens of a word attend to each other.
    current_word_id = -1
    sub_token_indices = []
    
    for i, w_id in enumerate(word_ids):
        if w_id is not None:
            if w_id != current_word_id and current_word_id != -1:
                if len(sub_token_indices) > 1:
                    for from_idx in sub_token_indices:
                        for to_idx in sub_token_indices:
                            predicted_matrix[from_idx, to_idx] = 1.0
                sub_token_indices = []
            
            sub_token_indices.append(i)
            current_word_id = w_id

    if len(sub_token_indices) > 1:
        for from_idx in sub_token_indices:
            for to_idx in sub_token_indices:
                predicted_matrix[from_idx, to_idx] = 1.0

    # Ensure special tokens have self-attention
    for i, w_id in enumerate(word_ids):
        if w_id is None:
            predicted_matrix[i, i] = 1.0
    
    # Normalize the matrix to represent a valid attention distribution
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    normalized_matrix = predicted_matrix / row_sums

    return "Punctuation-to-Next-Word Pattern", normalized_matrix