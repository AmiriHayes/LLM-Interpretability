import numpy as np

def conjunction_punctuation_attention_pattern(sentence, tokenizer):
    """
    Predicts the attention matrix for the 'Conjunction and Punctuation Attention' pattern.
    This pattern suggests that conjunctions (like 'and') and punctuation marks 
    (especially commas and periods) will attend to other punctuation or the end of the sentence.
    
    Args:
        sentence (str): The input sentence.
        tokenizer: A pre-trained tokenizer compatible with BERT.

    Returns:
        tuple: A tuple containing the name of the pattern and the predicted attention matrix.
    """
    tokens = tokenizer([sentence], return_tensors="pt")
    input_ids = tokens.input_ids[0]
    token_len = len(input_ids)
    
    # Initialize a zero matrix for attention weights
    predicted_matrix = np.zeros((token_len, token_len))

    # Get a list of the actual tokens as strings
    token_list = tokenizer.convert_ids_to_tokens(input_ids)

    # Identify indices of relevant tokens: commas, 'and', and the final period
    relevant_indices = []
    
    # Handle `CLS` token (always at index 0) and `SEP` token (always at the end)
    # Give them self-attention and attention to each other, as is common in BERT.
    predicted_matrix[0, 0] = 1.0
    predicted_matrix[-1, -1] = 1.0
    predicted_matrix[0, -1] = 1.0
    predicted_matrix[-1, 0] = 1.0
    
    # Get indices of commas, 'and', and the final period.
    # The final period is at `token_len - 1`
    final_period_idx = token_len - 1

    for i, token in enumerate(token_list):
        # We need to handle sub-tokens for conjunctions like "and" and "or".
        if token.lower() in [',', 'and', 'or', ':']:
            relevant_indices.append(i)

    # For each relevant punctuation or conjunction, give high attention to the final period
    # and to the tokens around it. This is a simplification of the observed behavior.
    for from_idx in relevant_indices:
        # Give high attention to the final period.
        if final_period_idx > 0:
            predicted_matrix[from_idx, final_period_idx] += 0.5
        
        # Give attention to adjacent punctuation or conjunctions.
        for to_idx in relevant_indices:
            if from_idx != to_idx and abs(from_idx - to_idx) <= 3:
                predicted_matrix[from_idx, to_idx] += 0.3
    
    # Normalize the matrix to ensure each row sums to 1.
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero for rows with no attention.
    predicted_matrix = np.divide(predicted_matrix, row_sums, out=np.zeros_like(predicted_matrix), where=row_sums != 0)
    
    return 'Conjunction and Punctuation Attention Pattern', predicted_matrix