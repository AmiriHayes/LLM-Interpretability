import numpy as np
import spacy
from typing import Tuple

# Load a spaCy model for dependency parsing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def sentence_boundary_alignment(sentence: str, tokenizer) -> Tuple[str, np.ndarray]:
    """
    Hypothesizes the attention pattern for Layer 6, Head 8, which is responsible
    for sentence boundary and punctuation alignment. It directs attention from
    various tokens, particularly those at the start or in key roles, to the
    final punctuation mark of the sentence.

    The function uses spaCy to identify the last token in a sentence (the punctuation)
    and creates a high-attention link from the first token and the last few tokens
    to this boundary marker. It also establishes a strong self-attention for punctuation marks.

    Args:
        sentence (str): The input sentence string.
        tokenizer: The tokenizer to use (e.g., AutoTokenizer).

    Returns:
        Tuple[str, np.ndarray]: A tuple containing the pattern name and the
                                predicted attention matrix of size (token_len, token_len).
    """
    doc = nlp(sentence)
    encoded_input = tokenizer(
        sentence,
        return_tensors="pt",
        add_special_tokens=True,
        return_offsets_mapping=True
    )
    token_len = len(encoded_input.input_ids[0])
    word_ids = encoded_input.word_ids(batch_index=0)
    
    predicted_matrix = np.zeros((token_len, token_len))

    # Find the indices of the final punctuation tokens
    final_punctuation_indices = []
    for sub_token_idx in range(token_len):
        if word_ids[sub_token_idx] is not None and doc[word_ids[sub_token_idx]].is_punct:
            if word_ids[sub_token_idx] == len(doc) - 1:
                final_punctuation_indices.append(sub_token_idx)
    
    # Identify the beginning of the sentence and key words
    initial_tokens_indices = []
    # Add the first token
    if token_len > 1:
        initial_tokens_indices.append(1)
        # Find the first few tokens of the sentence
        for i in range(1, min(4, token_len)):
            if word_ids[i] is not None:
                initial_tokens_indices.append(i)
    
    # Create the high-attention links to the final punctuation
    if final_punctuation_indices:
        for from_idx in initial_tokens_indices:
            for to_idx in final_punctuation_indices:
                predicted_matrix[from_idx, to_idx] = 1.0

        # Also add attention from the last few tokens to the final punctuation
        for from_idx in range(token_len - 4, token_len):
            for to_idx in final_punctuation_indices:
                predicted_matrix[from_idx, to_idx] = 1.0
                
    # Add self-attention for all punctuation tokens
    for i in range(token_len):
        if word_ids[i] is not None and doc[word_ids[i]].is_punct:
            predicted_matrix[i, i] = 1.0

    # Ensure self-attention for all tokens to create a valid distribution
    for i in range(token_len):
        if np.sum(predicted_matrix[i, :]) == 0:
            predicted_matrix[i, i] = 1.0

    # Normalize the matrix to represent a valid attention distribution
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    normalized_matrix = predicted_matrix / row_sums
    
    return "Sentence Boundary and Punctuation Alignment", normalized_matrix