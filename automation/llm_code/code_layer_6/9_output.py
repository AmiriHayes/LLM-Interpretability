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

def local_syntactic_attachment(sentence: str, tokenizer) -> Tuple[str, np.ndarray]:
    """
    Hypothesizes the attention pattern for Layer 6, Head 9, which is responsible
    for local syntactic modifier attachment. It creates a directed attention link
    from a token to its head in the dependency parse tree. This is a form of
    dependency parsing, but with a specific focus on local, immediate relationships
    rather than a full tree.

    The function uses spaCy's dependency parser to identify the head of each token.
    For each token, it then creates a high-attention link to its head token's
    position in the tokenized sequence.

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

    # Create a mapping from word_id to a list of subtoken indices
    word_to_subtoken_indices = {}
    for sub_token_idx, word_id in enumerate(word_ids):
        if word_id is not None:
            if word_id not in word_to_subtoken_indices:
                word_to_subtoken_indices[word_id] = []
            word_to_subtoken_indices[word_id].append(sub_token_idx)

    # Loop through each token in the spaCy document
    for token in doc:
        # Check if the token has a head (i.e., it's not the root)
        if token.head is not None and token.i != token.head.i:
            # Get the word indices for the current token and its head
            from_word_id = token.i
            to_word_id = token.head.i

            # Get the subtoken indices for the current token and its head
            if from_word_id in word_to_subtoken_indices and to_word_id in word_to_subtoken_indices:
                from_subtoken_indices = word_to_subtoken_indices[from_word_id]
                to_subtoken_indices = word_to_subtoken_indices[to_word_id]

                # Create attention links from all subtokens of the word to all subtokens of its head
                for from_idx in from_subtoken_indices:
                    for to_idx in to_subtoken_indices:
                        # Set attention from the token to its head
                        predicted_matrix[from_idx, to_idx] = 1.0

    # Handle special tokens [CLS] and [SEP].
    # [CLS] often attends to itself.
    predicted_matrix[0, 0] = 1.0
    # [SEP] often attends to [CLS] and itself.
    if token_len > 1:
        predicted_matrix[token_len - 1, 0] = 1.0
        predicted_matrix[token_len - 1, token_len - 1] = 1.0

    # Ensure self-attention for any token that doesn't have an attention link
    for i in range(token_len):
        if np.sum(predicted_matrix[i, :]) == 0:
            predicted_matrix[i, i] = 1.0

    # Normalize the matrix to represent a valid attention distribution
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    normalized_matrix = predicted_matrix / row_sums
    
    return "Local Syntactic Modifier Attachment", normalized_matrix