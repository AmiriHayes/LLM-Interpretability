import numpy as np

def leftward_local_attention(sentence, tokenizer):
    """
    Hypothesizes that this head performs a simple, leftward-looking
    attention pattern, where each token attends to the token immediately 
    preceding it.

    This function generates a rule-encoded attention matrix that mirrors 
    this pattern. Each token (from_token) at index i has high attention
    on the token (to_token) at index i-1. The [CLS] token has self-attention,
    and all tokens have a small amount of self-attention to account for 
    normalization.

    Args:
        sentence (str): The input sentence.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.

    Returns:
        tuple: A tuple containing the pattern name and the predicted attention matrix.
    """
    # Tokenize the sentence to get the length of the sequence
    tokens = tokenizer([sentence], return_tensors="pt")
    len_seq = len(tokens.input_ids[0])
    
    # Initialize the attention matrix with zeros
    predicted_matrix = np.zeros((len_seq, len_seq), dtype=float)

    # All tokens (from index 1 to len_seq-1) attend to the token immediately before them
    # The [CLS] token (index 0) has self-attention.
    for i in range(1, len_seq):
        predicted_matrix[i, i-1] = 1.0

    # Add self-attention for all tokens, which is necessary for uniform attention
    # to be created after normalization.
    for i in range(len_seq):
        predicted_matrix[i, i] = 1.0
        
    # The [CLS] token at index 0 should also attend to itself
    # and the first word.
    predicted_matrix[0, 0] = 1.0
    if len_seq > 1:
        predicted_matrix[0, 1] = 1.0

    # Normalize each row to ensure all values sum to 1.
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero for empty rows (if they exist)
    predicted_matrix = np.divide(predicted_matrix, row_sums, out=np.zeros_like(predicted_matrix), where=row_sums != 0)

    return 'Leftward Local Attention', predicted_matrix