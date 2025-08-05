import numpy as np
import collections

def repetition_and_parallelism_pattern(sentence, tokenizer):
    """
    Predicts attention patterns for Layer 1, Head 11 based on the Repetition/Parallelism Linking hypothesis.

    This function identifies identical tokens within a sentence (excluding common stop words)
    and predicts high attention weights between them. It also strongly links
    punctuation marks like quotation marks and question marks. The pattern is then
    encoded into a matrix.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer to process the sentence (e.g., from the transformers library).

    Returns:
        tuple: A tuple containing the pattern name and the predicted attention matrix.
    """
    stop_words = {'the', 'a', 'an', 'and', 'to', 'of', 'for', 'with', 'in', 'on', 'at'}
    
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    input_ids = toks.input_ids[0]
    word_ids = toks.word_ids()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Initialize the output matrix
    len_seq = len(tokens)
    predicted_matrix = np.zeros((len_seq, len_seq))

    # Add strong attention for special tokens [CLS] and [SEP]
    predicted_matrix[0, 0] = 1.0  # [CLS] self-attention
    predicted_matrix[-1, -1] = 1.0 # [SEP] self-attention

    # Find duplicate tokens and their indices
    token_indices = collections.defaultdict(list)
    for i, token in enumerate(tokens):
        # Exclude common stop words and special tokens from the repetition pattern
        # The model's attention data shows some linking of "of" and "a" but it's not a strong rule
        # and excluding them makes the pattern more specific and accurate to the data.
        if token.lower() not in stop_words and token not in ['[CLS]', '[SEP]']:
            token_indices[token].append(i)

    # Encode the repetition pattern
    for token, indices in token_indices.items():
        if len(indices) > 1:
            # Create a full-attention square between all instances of the repeated token
            for i in indices:
                for j in indices:
                    predicted_matrix[i, j] = 1.0
            
            # Additional logic for punctuation like quotation marks
            if token in ["'", '"', '?', '!']:
                for i in indices:
                    for j in indices:
                        if i != j:
                            # Punctuation links are often very strong and bidirectional
                            predicted_matrix[i, j] = 1.0
                            predicted_matrix[j, i] = 1.0
            else:
                 # Bidirectional attention between all matching tokens
                 for i in indices:
                    for j in indices:
                        if i != j:
                            predicted_matrix[i, j] = 1.0
                            predicted_matrix[j, i] = 1.0

    # Normalize each row to sum to 1 to simulate attention's softmax function
    # This assumes uniform attention to all linked tokens
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero for rows with no attention
    normalized_matrix = predicted_matrix / row_sums
    
    return "Repetition/Parallelism Linking Pattern", normalized_matrix