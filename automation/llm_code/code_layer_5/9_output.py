import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizer

def adjacent_rightward_attention(sentence: str, tokenizer: PreTrainedTokenizer) -> Tuple[str, np.ndarray]:
    """
    Hypothesizes the attention pattern for Layer 5, Head 9 as a 'Direct Rightward Adjacency' pattern.

    This function creates a rule-encoded attention matrix where each token
    attends to the token immediately to its right. This pattern reflects a
    simple, sequential, token-level cohesion mechanism.

    Parameters:
    - sentence (str): The input sentence.
    - tokenizer: A pre-trained BERT tokenizer instance.

    Returns:
    - tuple: A tuple containing the name of the pattern and the predicted
             attention matrix.
    """
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    predicted_matrix = np.zeros((len_seq, len_seq))

    # All tokens attend to the next token to their right
    for i in range(len_seq - 1):
        predicted_matrix[i, i + 1] = 1.0

    # Special tokens [CLS] and [SEP] still get self-attention as a default
    # and to ensure a non-zero row sum for normalization if they don't attend to anything else.
    if len_seq > 0:
        predicted_matrix[0, 0] = 1.0  # [CLS] self-attention
    if len_seq > 1:
        predicted_matrix[-1, -1] = 1.0 # [SEP] self-attention

    # Normalize the matrix rows so each row sums to 1.
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    normalized_matrix = predicted_matrix / row_sums

    return 'Direct Rightward Adjacency Pattern', normalized_matrix