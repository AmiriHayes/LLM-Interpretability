import numpy as np
import torch

def end_of_sentence_pattern(sentence, tokenizer):
    """
    Hypothesizes a 'End-of-Sentence Attention' pattern for a given sentence.

    This function predicts a matrix where all tokens attend to the final
    punctuation mark (period, question mark, or exclamation point) of the
    sentence. It normalizes the attention weights by row to approximate a
    uniform distribution of attention to the final token.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer object (e.g., from Hugging Face).

    Returns:
        tuple: A tuple containing the pattern name and the predicted attention matrix.
               The matrix is of size (token_len x token_len).
    """

    # Tokenize the sentence and get the length of the token sequence
    toks = tokenizer([sentence], return_tensors="pt", add_special_tokens=True)
    token_ids = toks.input_ids[0]
    token_len = len(token_ids)

    # Initialize a matrix of zeros
    predicted_matrix = np.zeros((token_len, token_len))

    # Identify the index of the last token.
    # The last token is typically a punctuation mark.
    final_token_index = token_len - 1

    # For each token in the sentence (excluding the final token),
    # assign an attention weight of 1.0 to the final token.
    for i in range(token_len):
        predicted_matrix[i, final_token_index] = 1.0

    # Handle special tokens. The CLS and EOS tokens should attend to themselves
    # as per BERT's typical attention patterns. In this model, they also
    # attend to the final token. Let's make sure the special tokens have
    # some self-attention while also maintaining the overall pattern.
    predicted_matrix[0, 0] = 1.0  # CLS token self-attention
    
    # The final token (punctuation) should also have self-attention.
    predicted_matrix[final_token_index, final_token_index] = 1.0
    
    # Normalize the matrix rows so the sum of each row is 1.
    # This simulates a uniform distribution of attention to the final token.
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero for any rows that might sum to zero, although
    # this is unlikely with the current logic.
    row_sums[row_sums == 0] = 1
    predicted_matrix = predicted_matrix / row_sums

    return 'End-of-Sentence Attention', predicted_matrix

# Example of how to use the function:
# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# sentence = "The sun dipped below the horizon."
# pattern_name, matrix = end_of_sentence_pattern(sentence, tokenizer)
# print(f"Pattern: {pattern_name}")
# print("Predicted Attention Matrix:")
# print(matrix)