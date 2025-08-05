import numpy as np
from typing import Tuple

def consecutive_token_linking(sentence: str, tokenizer) -> Tuple[str, np.ndarray]:
    """
    Hypothesizes a 'Consecutive Token Linking' attention model.

    This pattern assumes attention flows from each token to the token immediately
    following it, creating a strong forward-looking link.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer object (e.g., BertTokenizer).

    Returns:
        tuple[str, np.ndarray]: A tuple containing the pattern name and the
                                predicted attention matrix.
    """
    # Tokenize the sentence to get the sequence length
    tokens = tokenizer([sentence], return_tensors="np", max_length=512, truncation=True)
    seq_len = len(tokens['input_ids'][0])

    # Initialize the predicted attention matrix with zeros
    predicted_matrix = np.zeros((seq_len, seq_len))

    # Assign a high attention weight from each token to the next token
    # This creates a strong forward-pointing link
    for i in range(seq_len - 1):
        predicted_matrix[i, i+1] = 1.0

    # Assign some self-attention for the special tokens [CLS] and [SEP]
    # This is a common pattern in many attention heads.
    if seq_len > 0:
        predicted_matrix[0, 0] = 0.5 # [CLS] token
    if seq_len > 1:
        predicted_matrix[-1, -1] = 0.5 # [SEP] token
    
    # Normalize each row to sum to 1.0
    for i in range(seq_len):
        row_sum = np.sum(predicted_matrix[i, :])
        if row_sum > 0:
            predicted_matrix[i, :] /= row_sum
    
    return 'Consecutive Token Linking Pattern', predicted_matrix