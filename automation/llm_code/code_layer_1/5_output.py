import numpy as np
import spacy
from typing import Tuple
from transformers import AutoTokenizer

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

def parenthetical_attachment_pattern(sentence: str, tokenizer) -> Tuple[str, np.ndarray]:
    """
    Hypothesizes the attention pattern for Layer 1, Head 5 based on parenthetical
    or subordinate clause attachment, with improved efficiency.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer object (e.g., from Hugging Face).

    Returns:
        Tuple[str, np.ndarray]: A tuple containing the pattern name and a
        predicted attention matrix of shape (token_len, token_len).
    """

    # Use a fast tokenizer for efficient tokenization and offset mapping
    tokens = tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True)
    input_ids = tokens.input_ids[0].tolist()
    len_seq = len(input_ids)
    offset_mapping = tokens.offset_mapping[0].tolist()

    # Initialize an attention matrix with zeros
    predicted_matrix = np.zeros((len_seq, len_seq), dtype=np.float32)

    # Add self-attention for CLS and SEP tokens
    predicted_matrix[0, 0] = 1.0  # CLS token
    predicted_matrix[len_seq - 1, len_seq - 1] = 1.0  # SEP token

    doc = nlp(sentence)
    
    # Map spaCy tokens to tokenizer indices for efficient lookup
    spacy_to_tokenizer_map = {}
    for spacy_token in doc:
        char_start, char_end = spacy_token.idx, spacy_token.idx + len(spacy_token.text)
        # Find the tokenizer token that corresponds to the spaCy token
        for i, (offset_start, offset_end) in enumerate(offset_mapping):
            if offset_start == char_start and offset_end == char_end:
                spacy_to_tokenizer_map[spacy_token.i] = i
                break

    # Identify and handle clauses and punctuation
    for token in doc:
        token_idx = spacy_to_tokenizer_map.get(token.i)
        if token_idx is None:
            continue

        # Handle subordinate clauses and parenthetical phrases
        if token.pos_ == "SCONJ" or token.text in [",", "but", "and"]:
            # Direct attention to the first token of the main clause (index 1 for BERT)
            predicted_matrix[token_idx, 1] = 1.0

        # High self-attention for commas and quotation marks
        if token.text in [",", "'", '"']:
            predicted_matrix[token_idx, token_idx] = 1.0

        # Strong attachment from a trailing question mark to the start of the clause
        if token.text == "?":
            # Find the start of the clause/question
            start_of_clause_idx = 1  # Default to the first token
            for prev_token in doc[:token.i]:
                if prev_token.text in [",", ":", ";"]:
                    start_of_clause_idx = spacy_to_tokenizer_map.get(prev_token.i + 1, start_of_clause_idx)
                    break
            predicted_matrix[token_idx, start_of_clause_idx] = 1.0

    # Normalize the matrix rows so they sum to 1, vectorized
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    normalized_matrix = predicted_matrix / row_sums

    return "Parenthetical Attachment Pattern", normalized_matrix