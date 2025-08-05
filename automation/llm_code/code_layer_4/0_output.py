import numpy as np
from transformers import BertTokenizer

def compound_word_pattern(sentence: str, tokenizer) -> tuple[str, np.ndarray]:
    """
    Hypothesizes a 'Compound Word and Morphological Awareness' pattern for a
    given sentence.

    This pattern is characterized by high attention from a word-piece token
    (one starting with '##') to the preceding word-piece tokens that form the same word.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer object (e.g., BertTokenizer).

    Returns:
        tuple[str, np.ndarray]: A tuple containing the pattern name and the
                                predicted attention matrix.
    """
    toks = tokenizer(sentence, return_tensors="np")
    input_ids = toks["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    seq_len = len(tokens)
    predicted_matrix = np.zeros((seq_len, seq_len))

    # Identify CLS and SEP tokens
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id

    # Get word IDs for each token
    word_ids = toks.word_ids()

    # Iterate through tokens to build the attention pattern
    for i in range(1, seq_len):
        current_token = tokens[i]
        
        # If a token is a word-piece (starts with '##'), attend to the
        # preceding token that is part of the same word.
        if current_token.startswith("##") and word_ids[i] == word_ids[i-1]:
            # Assign high attention from the word-piece to its preceding part.
            # This is a simplification; in reality, it may look back multiple
            # tokens to assemble the full word. We'll add a weight to all
            # preceding tokens that are part of the same word.
            predicted_matrix[i, i-1] = 1.0
        
        # All tokens should have some attention to themselves
        predicted_matrix[i, i] = 1.0

    # Handle special tokens (CLS and SEP)
    if tokens[0] == tokenizer.cls_token:
        predicted_matrix[0, 0] = 1.0
    if tokens[-1] == tokenizer.sep_token:
        predicted_matrix[-1, -1] = 1.0
        
    # Normalize each row to sum to 1.0 (uniform attention distribution)
    # This ensures the matrix is a valid attention matrix.
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    predicted_matrix = np.divide(predicted_matrix, row_sums, out=np.zeros_like(predicted_matrix), where=row_sums!=0)

    return 'Compound Word and Morphological Awareness Pattern', predicted_matrix

# Example usage:
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# sentence = "The old, creaky house stood on the hill."
# pattern_name, matrix = compound_word_pattern(sentence, tokenizer)
# print(pattern_name)
# print(matrix)