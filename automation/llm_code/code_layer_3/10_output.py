import numpy as np

def subword_to_prefix(sentence, tokenizer):
    """
    Hypothesizes that this head links a subword token (starting with '##')
    to the token immediately preceding it.

    This function generates a predicted attention matrix where any token starting
    with '##' gives its full attention to the token directly before it. This models
    the head's role in re-assembling fragmented words from the tokenizer.

    Args:
        sentence (str): The input sentence.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.

    Returns:
        tuple: A tuple containing the pattern name and the predicted attention matrix.
    """
    tokenized_sentence = tokenizer.tokenize(sentence)
    tokenized_sentence = ['[CLS]'] + tokenized_sentence + ['[SEP]']
    len_seq = len(tokenized_sentence)
    predicted_matrix = np.zeros((len_seq, len_seq), dtype=float)

    for i in range(len_seq):
        from_token = tokenized_sentence[i]

        # Check if the token is a subword token (starts with '##')
        if from_token.startswith('##') and i > 0:
            # If it is, it attends to the previous token
            predicted_matrix[i, i-1] = 1.0
        else:
            # All other tokens attend to themselves for simplicity
            predicted_matrix[i, i] = 1.0

    # Normalize each row to ensure the attention weights sum to 1.
    # This also handles cases where a token doesn't attend to anything,
    # as its row sum would be 0, and dividing by 0 is handled by out=np.zeros_like.
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    predicted_matrix = np.divide(predicted_matrix, row_sums, out=np.zeros_like(predicted_matrix), where=row_sums != 0)

    return 'Subword to Prefix Pattern', predicted_matrix