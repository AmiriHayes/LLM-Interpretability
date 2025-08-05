import numpy as np

def right_to_left_local_dependencies(sentence: str, tokenizer) -> tuple[str, np.ndarray]:
    """
    Hypothesizes a 'Right-to-Left Local Dependency Parsing' attention pattern.

    This pattern is characterized by tokens having high attention to the token
    immediately preceding them, capturing local dependencies, compound words,
    and phrase structure.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer object (e.g., BertTokenizer).

    Returns:
        tuple[str, np.ndarray]: A tuple containing the pattern name and the
                                predicted attention matrix.
    """
    toks = tokenizer([sentence], return_tensors="np", add_special_tokens=True)
    seq_len = len(toks["input_ids"][0])
    
    # Initialize matrix with low, uniform attention
    predicted_matrix = np.full((seq_len, seq_len), 0.1)

    # Assign high attention from each token to the one immediately to its left
    # This loop starts at the second token (index 2) to skip the [CLS] token
    # and assigns a high weight to the token at index j-1.
    for i in range(1, seq_len):
        predicted_matrix[i, i-1] = 0.8
        
    # Special tokens [CLS] and [SEP] typically have high self-attention
    if seq_len > 0:
        predicted_matrix[0, 0] = 1.0  # [CLS] token
    if seq_len > 1:
        predicted_matrix[seq_len - 1, seq_len - 1] = 1.0  # [SEP] token
    
    # Normalize each row to ensure they sum to 1.0.
    # We must do this carefully to not mess up our special token weights
    for i in range(seq_len):
        row_sum = np.sum(predicted_matrix[i, :])
        if row_sum > 0:
            predicted_matrix[i, :] /= row_sum
    
    return 'Right-to-Left Local Dependency Parsing', predicted_matrix

# Example usage:
# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# sentence = "The old, creaky house, standing on the hill, seemed to whisper secrets of times long past, didn't it?"
# pattern_name, matrix = right_to_left_local_dependencies(sentence, tokenizer)
# print(pattern_name)
# print(matrix)