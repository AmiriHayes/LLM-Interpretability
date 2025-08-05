import numpy as np

def local_word_piece_merging(sentence, tokenizer):
    """
    Predicts an attention matrix for the 'Word-Piece Merging Pattern'.
    
    This function hypothesizes that Layer 2, Head 1 is responsible for
    merging subword tokens back into full words. The pattern is defined
    by high attention weights from the end of a subword token to the
    beginning of the same word.
    
    Args:
        sentence (str): The input sentence.
        tokenizer: A tokenizer with `tokenize` and `convert_tokens_to_ids` methods.
        
    Returns:
        tuple: A tuple containing the pattern name and the predicted attention matrix.
    """
    
    # Tokenize the sentence to get word pieces
    tokens = tokenizer.tokenize(sentence)
    
    # Add special tokens for BERT
    full_tokens = ['[CLS]'] + tokens + ['[SEP]']
    
    token_len = len(full_tokens)
    predicted_matrix = np.zeros((token_len, token_len))
    
    # Loop through tokens and identify word-piece connections
    for i in range(1, token_len):
        current_token = full_tokens[i]
        
        # Check if the current token is a subword piece (starts with '##')
        if current_token.startswith('##'):
            # The pattern is an attention from the subword piece to the token that starts the word
            # We assume a word is made up of consecutive tokens and the subword piece
            # will always attend to the first token of the word it is a part of.
            # We can find this by moving backwards until we find a token that doesn't start with '##'.
            
            # Find the start of the word
            for j in range(i - 1, 0, -1):
                if not full_tokens[j].startswith('##'):
                    predicted_matrix[i, j] = 1.0
                    break
    
    # Add self-attention for the [CLS] token and [SEP] token
    # While the provided examples don't show this explicitly, it's a common pattern for special tokens
    # and a sensible generalization to avoid a zero-attention row.
    predicted_matrix[0, 0] = 1.0
    predicted_matrix[-1, -1] = 1.0

    # Normalize each row to ensure the attention weights sum to 1
    # This is a key part of how attention works
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    predicted_matrix = np.divide(predicted_matrix, row_sums, out=np.zeros_like(predicted_matrix), where=row_sums != 0)
    
    return 'Word-Piece Merging Pattern', predicted_matrix