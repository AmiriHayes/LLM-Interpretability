import torch
import numpy as np
from transformers import BertTokenizer

def subword_morpheme_alignment(sentence: str, tokenizer: BertTokenizer) -> tuple[str, np.ndarray]:
    """
    Predicts the attention pattern for Layer 1, Head 4, based on the hypothesis
    that it aligns subword tokens with their preceding morphemes.
    
    Parameters:
    sentence (str): The input sentence.
    tokenizer (BertTokenizer): The BERT tokenizer for tokenizing the sentence.
    
    Returns:
    tuple[str, np.ndarray]: The name of the pattern and the predicted attention matrix.
    """
    
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    input_ids = toks.input_ids.squeeze()
    
    # Get the tokenized words and their IDs
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Determine the sequence length and initialize the attention matrix
    len_seq = len(tokens)
    predicted_matrix = np.zeros((len_seq, len_seq))

    # Add self-attention for the [CLS] token and the final [SEP] token
    predicted_matrix[0, 0] = 1
    predicted_matrix[-1, -1] = 1

    # Loop through tokens to identify subword units and assign attention
    for i in range(1, len_seq - 1):
        current_token = tokens[i]
        
        # Check if the token is a subword piece (starts with '##')
        if current_token.startswith('##'):
            # If it's a subword, assign attention to the immediately preceding token
            predicted_matrix[i, i-1] = 1
        else:
            # For other tokens, assign self-attention
            predicted_matrix[i, i] = 1

    # Normalize each row to ensure the attention weights sum to 1
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    # Handle the case of zero-sum rows to avoid division by zero
    row_sums[row_sums == 0] = 1
    normalized_matrix = predicted_matrix / row_sums
    
    return 'Subword Morpheme Alignment', normalized_matrix