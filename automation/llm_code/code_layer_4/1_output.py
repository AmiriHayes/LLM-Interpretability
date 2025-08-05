import numpy as np
from transformers import BertTokenizer

def start_of_clause_focus_pattern(sentence: str, tokenizer) -> tuple[str, np.ndarray]:
    """
    Hypothesizes a 'Start of Sentence/Clause Focus' attention pattern.

    This pattern is characterized by all tokens attending to the first token
    of the sentence or clause, with some self-attention and general
    attention to other tokens.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer object (e.g., BertTokenizer).

    Returns:
        tuple[str, np.ndarray]: A tuple containing the pattern name and the
                                predicted attention matrix.
    """
    toks = tokenizer(sentence, return_tensors="np")
    input_ids = toks["input_ids"][0]
    seq_len = len(input_ids)
    
    # Initialize a matrix with a base attention value for all tokens.
    predicted_matrix = np.full((seq_len, seq_len), 0.1)

    # All tokens strongly attend to the first token of the sentence.
    for i in range(seq_len):
        predicted_matrix[i, 1] = 0.5  # Assuming index 1 is the first non-special token.
        
    # All tokens also have some self-attention.
    np.fill_diagonal(predicted_matrix, 0.4)

    # Set self-attention for special tokens CLS and SEP.
    predicted_matrix[0, 0] = 1.0
    predicted_matrix[seq_len - 1, seq_len - 1] = 1.0

    # Normalize each row to sum to 1.0.
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    predicted_matrix = np.divide(predicted_matrix, row_sums, out=np.zeros_like(predicted_matrix), where=row_sums!=0)

    return 'Start of Sentence/Clause Focus Pattern', predicted_matrix

# Example usage:
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# sentence = "The old, creaky house stood on the hill."
# pattern_name, matrix = start_of_clause_focus_pattern(sentence, tokenizer)
# print(pattern_name)
# print(matrix)