import numpy as np
from transformers import BertTokenizer

def local_context_attention(sentence: str, tokenizer: BertTokenizer) -> tuple[str, np.ndarray]:
    """
    Predicts the attention matrix for Layer 2, Head 5, based on the hypothesis
    that it focuses on simple local context (self + immediate neighbors).

    Each token attends to itself, its direct left neighbor (if exists),
    and its direct right neighbor (if exists). The attention is uniformly
    distributed among these neighbors.

    Args:
        sentence (str): The input sentence.
        tokenizer (BertTokenizer): The tokenizer for the model.

    Returns:
        tuple[str, np.ndarray]: A tuple containing the name of the pattern
                                and the predicted attention matrix.
    """
    # Tokenize the sentence to get input IDs and sequence length
    tokens = tokenizer([sentence], return_tensors="pt")
    input_ids = tokens['input_ids'][0]
    seq_len = len(input_ids)
    
    # Initialize a zero matrix for attention weights
    predicted_matrix = np.zeros((seq_len, seq_len), dtype=np.float32)

    # Apply the local context attention pattern
    for i in range(seq_len):
        # Current token attends to itself
        predicted_matrix[i, i] = 1.0
        
        # Current token attends to its left neighbor (if it exists and is not CLS)
        if i > 0:
            predicted_matrix[i, i - 1] = 1.0
        
        # Current token attends to its right neighbor (if it exists and is not SEP)
        if i < seq_len - 1:
            predicted_matrix[i, i + 1] = 1.0
            
    # Normalize each row to sum to 1 to represent a probability distribution.
    # This ensures that the attention weights for each token sum up to 1.
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero for rows that might have no attention targets (shouldn't happen here)
    row_sums[row_sums == 0] = 1.0
    normalized_matrix = predicted_matrix / row_sums

    return "Simple Local Context Attention", normalized_matrix

# Example usage (for demonstration, not part of the required function)
if __name__ == '__main__':
    # Initialize a BERT tokenizer (you would typically load this from a pre-trained model)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    test_sentence_1 = "The quick brown fox jumps over the lazy dog."
    pattern_name_1, pred_matrix_1 = local_context_attention(test_sentence_1, tokenizer)

    print(f"Pattern Name: {pattern_name_1}")
    print("Predicted Attention Matrix Shape:", pred_matrix_1.shape)
    print("\nSample Attention for 'quick' (token 2):")
    # Assuming 'quick' is token 2 (after [CLS] and 'The')
    # It should attend to [CLS], 'The', 'quick', 'brown'
    # The actual indices depend on the tokenizer's output for the sentence.
    # Let's print the row for a specific token, e.g., the token 'quick'
    
    # Find the token ID for 'quick' in the tokenized sentence
    tokens_list = tokenizer.convert_ids_to_tokens(tokenizer.encode(test_sentence_1, add_special_tokens=True))
    try:
        quick_idx = tokens_list.index('quick')
        print(f"Attention from 'quick' (token {quick_idx}): {pred_matrix_1[quick_idx, :]}")
    except ValueError:
        print("Token 'quick' not found in tokenized sentence.")

    test_sentence_2 = "Hello world!"
    pattern_name_2, pred_matrix_2 = local_context_attention(test_sentence_2, tokenizer)
    print(f"\nPattern Name: {pattern_name_2}")
    print("Predicted Attention Matrix Shape:", pred_matrix_2.shape)
    
