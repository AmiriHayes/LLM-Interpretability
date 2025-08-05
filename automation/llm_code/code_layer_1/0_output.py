import numpy as np
import spacy
from transformers import PreTrainedTokenizer

def clause_boundary_pattern(sentence: str, tokenizer: PreTrainedTokenizer):
    """
    Hypothesizes a 'Clause Boundary Pattern' attention matrix.

    This function predicts a matrix where each token's attention is focused on the
    final punctuation mark of the sentence, specifically the period (.), question mark (?),
    or exclamation mark (!). This simulates a pattern of attending to the sentence's
    end. Additionally, it assigns self-attention for special tokens [CLS] and [SEP].
    
    Args:
        sentence (str): The input sentence.
        tokenizer (PreTrainedTokenizer): The tokenizer used to process the sentence.

    Returns:
        tuple[str, np.ndarray]: A tuple containing the pattern name and the predicted
                                attention matrix. The matrix is a square numpy array
                                with dimensions (token_len * token_len).
    """

    toks = tokenizer([sentence], return_tensors="pt")
    input_ids = toks.input_ids[0]
    token_len = len(input_ids)
    
    # Initialize an attention matrix of zeros
    predicted_matrix = np.zeros((token_len, token_len))

    # Find the index of the final punctuation mark
    # We check for period, question mark, or exclamation mark as the last token
    # or the token right before the [SEP] token
    final_punc_idx = -1
    for i in range(token_len - 1, 0, -1):
        token = tokenizer.decode(input_ids[i])
        if token in ['.', '?', '!', "'"] or tokenizer.decode(input_ids[i-1]) in ['.', '?', '!', "'"]:
            final_punc_idx = i
            break
            
    # If a punctuation mark is not found, default to the last non-SEP token
    if final_punc_idx == -1:
        final_punc_idx = token_len - 2

    # All tokens (excluding CLS and SEP) attend to the final punctuation.
    for i in range(1, token_len - 1):
        predicted_matrix[i, final_punc_idx] = 1.0

    # CLS and SEP tokens have self-attention.
    # Note: A self-attention pattern for CLS and SEP is a common baseline.
    predicted_matrix[0, 0] = 1.0
    predicted_matrix[token_len - 1, token_len - 1] = 1.0

    # Normalize each row to ensure the attention weights for each token sum to 1.
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero for rows that are entirely zero
    predicted_matrix = np.divide(predicted_matrix, row_sums, out=np.zeros_like(predicted_matrix), where=row_sums!=0)

    return 'Clause Boundary Pattern', predicted_matrix