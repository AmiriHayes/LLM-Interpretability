import numpy as np

def next_token_prediction(sentence, tokenizer):
    """
    Hypothesizes that this head links each token to the next token in the sequence.

    The function generates a predicted attention matrix where each token
    (from_token) gives its full attention to the token immediately following it
    (to_token). Punctuation is treated like any other token. This represents
    a simple, sequential linguistic pattern.

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

    # The pattern is simple: each token looks at the next token.
    # The last token, [SEP], has nowhere to look, so we can give it self-attention.
    for i in range(len_seq - 1):
        predicted_matrix[i, i + 1] = 1.0

    # Add attention for the final token and CLS token as they have no "next" token
    # or follow a different pattern. For simplicity, we can have them attend to themselves
    # or the end of the sequence, or simply have a uniform distribution for them.
    # We will just normalize rows, which will handle the final token having a sum of 0
    # and resulting in a uniform distribution for it.
    predicted_matrix[len_seq - 1, len_seq - 1] = 1.0
    predicted_matrix[0, 0] = 1.0

    # Normalize each row to ensure the attention weights for each from_token sum to 1.
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    predicted_matrix = np.divide(predicted_matrix, row_sums, out=np.zeros_like(predicted_matrix), where=row_sums != 0)

    return 'Next Token Prediction Pattern', predicted_matrix