import numpy as np

def sub_word_alignment(sentence, tokenizer):
    """
    Predicts an attention matrix for a head responsible for Sub-Word Token Alignment.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer object with `word_ids` capabilities (e.g., from Hugging Face).

    Returns:
        tuple: A tuple containing the name of the pattern and the predicted attention matrix.
    """
    toks = tokenizer(sentence, return_tensors="pt")
    input_ids = toks.input_ids[0]
    token_len = len(input_ids)
    word_ids = toks.word_ids(batch_index=0)
    
    # Initialize an empty attention matrix
    predicted_matrix = np.zeros((token_len, token_len))

    # Identify tokens that are part of the same word
    word_groups = {}
    for i, word_id in enumerate(word_ids):
        if word_id is not None:
            if word_id not in word_groups:
                word_groups[word_id] = []
            word_groups[word_id].append(i)

    # Assign high attention between sub-word tokens
    for group in word_groups.values():
        if len(group) > 1:
            for from_token_index in group:
                for to_token_index in group:
                    if from_token_index != to_token_index:
                        # Assign attention from a token to all other tokens in the same word
                        predicted_matrix[from_token_index, to_token_index] = 1.0

    # For any token that is part of a word, it should also attend to itself
    for i in range(token_len):
        if word_ids[i] is not None:
            predicted_matrix[i, i] = 1.0
    
    # Handle special tokens. [CLS] and [SEP] attend to themselves.
    # The tokenizer's word_ids will be None for these tokens.
    if word_ids[0] is None:  # CLS token
        predicted_matrix[0, 0] = 1.0
    if word_ids[-1] is None: # SEP token
        predicted_matrix[-1, -1] = 1.0
        
    # Normalize the matrix by row to ensure each row sums to 1
    # This simulates a softmax distribution
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero for rows that are all zeros
    predicted_matrix = np.divide(predicted_matrix, row_sums, out=np.zeros_like(predicted_matrix), where=row_sums != 0)
    
    return 'Sub-Word Token Alignment', predicted_matrix