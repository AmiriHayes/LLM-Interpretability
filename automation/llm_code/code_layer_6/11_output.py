import numpy as np
from typing import Tuple

def local_contextual_linking(sentence: str, tokenizer) -> Tuple[str, np.ndarray]:
    """
    Hypothesizes the attention pattern for Layer 6, Head 11, which is responsible
    for linking each token to the token immediately preceding it. This pattern
    models local, short-range dependencies, capturing the sequential flow of
    information in a sentence.

    Args:
        sentence (str): The input sentence string.
        tokenizer: The tokenizer to use (e.g., AutoTokenizer).

    Returns:
        Tuple[str, np.ndarray]: A tuple containing the pattern name and the
                                predicted attention matrix of size (token_len, token_len).
    """
    # Tokenize the sentence and get input IDs
    encoded_input = tokenizer(
        sentence,
        return_tensors="pt",
        add_special_tokens=True
    )
    token_len = len(encoded_input.input_ids[0])
    
    # Initialize a square matrix of zeros
    predicted_matrix = np.zeros((token_len, token_len))

    # The CLS token should have self-attention
    predicted_matrix[0, 0] = 1.0

    # Every token after the CLS token and before the EOS token attends to the
    # token directly preceding it.
    for i in range(1, token_len):
        predicted_matrix[i, i-1] = 1.0
    
    # Normalize the matrix to represent a valid attention distribution
    # This pattern naturally sums to 1.0 per row (for i > 0), but let's
    # explicitly normalize to be robust.
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    normalized_matrix = predicted_matrix / row_sums

    return "Local Contextual Linking", normalized_matrix