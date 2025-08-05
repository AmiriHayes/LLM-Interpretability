import numpy as np
from typing import List, Tuple
from transformers import AutoTokenizer

def compound_word_unification(sentence: str, tokenizer: AutoTokenizer) -> Tuple[str, np.ndarray]:
    """
    Hypothesizes the attention pattern for Layer 6, Head 4.
    This function identifies sub-tokens that form a single word and
    assigns attention between them. It also identifies certain
    common compound words and assigns attention between their parts.
    
    Args:
        sentence (str): The input sentence string.
        tokenizer: The tokenizer to use (e.g., from `transformers`).
    
    Returns:
        Tuple[str, np.ndarray]: A tuple containing the pattern name and the
                                predicted attention matrix of size (token_len, token_len).
    """
    # Tokenize the sentence and get word IDs
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

    # Helper function to get all token indices for a given word ID
    def get_indices_for_word_id(w_id):
        return [i for i, x in enumerate(word_ids) if x == w_id]

    # Assign self-attention for special tokens
    for i, w_id in enumerate(word_ids):
        if w_id is None:
            predicted_matrix[i, i] = 1.0

    # Iterate through each word to find sub-token relationships
    current_word_id = -1
    sub_token_indices = []
    
    for i, w_id in enumerate(word_ids):
        if w_id is not None:
            if w_id != current_word_id and current_word_id != -1:
                # New word detected, process the previous one
                if len(sub_token_indices) > 1:
                    # Create a fully connected sub-matrix for this word's tokens
                    for from_idx in sub_token_indices:
                        for to_idx in sub_token_indices:
                            predicted_matrix[from_idx, to_idx] = 1.0
                sub_token_indices = []
            
            sub_token_indices.append(i)
            current_word_id = w_id

    # Process the last word after the loop ends
    if len(sub_token_indices) > 1:
        for from_idx in sub_token_indices:
            for to_idx in sub_token_indices:
                predicted_matrix[from_idx, to_idx] = 1.0

    # Normalize the matrix to represent a valid attention distribution
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # Prevent division by zero for rows with no attention
    normalized_matrix = predicted_matrix / row_sums

    return "Compound Word and Phrase Unification Pattern", normalized_matrix