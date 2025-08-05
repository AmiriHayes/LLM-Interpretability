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

def parenthetical_appositive_linking(sentence: str, tokenizer) -> Tuple[str, np.ndarray]:
    """
    Hypothesizes the attention pattern for Layer 6, Head 10, which is responsible
    for linking parenthetical and appositive phrases to the word or phrase they modify.
    The function identifies appositive modifiers and parenthetical clauses (typically
    marked by commas) and directs attention from the start of the modifier/clause
    to the word it is modifying.

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

    # Map word_ids to subtoken indices for aligning spaCy tokens with BERT tokens
    word_to_subtoken_indices = {}
    for sub_token_idx, word_id in enumerate(word_ids):
        if word_id is not None:
            if word_id not in word_to_subtoken_indices:
                word_to_subtoken_indices[word_id] = []
            word_to_subtoken_indices[word_id].append(sub_token_idx)

    # Loop through spaCy tokens to find appositives and parentheticals
    for token in doc:
        # Check for appositive modifiers
        if token.dep_ == 'appos':
            # The appositive token (or its first subtoken) attends to its head (the modified noun)
            from_word_id = token.i
            to_word_id = token.head.i
            if from_word_id in word_to_subtoken_indices and to_word_id in word_to_subtoken_indices:
                from_idx = word_to_subtoken_indices[from_word_id][0]  # Take the first subtoken
                to_idx = word_to_subtoken_indices[to_word_id][0]
                predicted_matrix[from_idx, to_idx] = 1.0

        # Check for punctuation that introduces a parenthetical
        if token.text == ',' and token.head.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ']:
            # The comma token attends to the word it is modifying
            from_word_id = token.i
            to_word_id = token.head.i
            if from_word_id in word_to_subtoken_indices and to_word_id in word_to_subtoken_indices:
                from_subtoken_indices = word_to_subtoken_indices[from_word_id]
                to_subtoken_indices = word_to_subtoken_indices[to_word_id]
                # High attention from the comma to the first subtoken of its head
                for from_idx in from_subtoken_indices:
                    predicted_matrix[from_idx, to_subtoken_indices[0]] = 1.0

    # Ensure self-attention for any token that does not have an attention link
    for i in range(token_len):
        if np.sum(predicted_matrix[i, :]) == 0:
            predicted_matrix[i, i] = 1.0
    
    # Normalize the matrix to represent a valid attention distribution
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    normalized_matrix = predicted_matrix / row_sums
    
    return "Parenthetical and Appositive Linking", normalized_matrix