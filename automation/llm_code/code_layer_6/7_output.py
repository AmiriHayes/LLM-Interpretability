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

def subject_verb_alignment(sentence: str, tokenizer) -> Tuple[str, np.ndarray]:
    """
    Hypothesizes the attention pattern for Layer 6, Head 7, which is responsible
    for long-distance subject-verb alignment. It connects a sentence's subject
    (and its modifiers) to the main verb, especially when a parenthetical phrase
    or other structural interruption is present.

    The function uses spaCy to identify the subject and verb and predicts a strong
    attention link from subject-related tokens to the verb.

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

    # Identify the main verb and its subject using spaCy dependency parsing
    verb_indices = []
    subject_indices = []

    for token in doc:
        if "VERB" in token.pos_ or "AUX" in token.pos_:
            for sub_token_idx, w_id in enumerate(word_ids):
                if w_id is not None and doc[w_id] == token:
                    verb_indices.append(sub_token_idx)
            
            # Find the subject (nsubj) of the verb
            for child in token.children:
                if "nsubj" in child.dep_:
                    for sub_token_idx, w_id in enumerate(word_ids):
                        if w_id is not None and doc[w_id] == child:
                            subject_indices.append(sub_token_idx)
                    # Also find any modifiers of the subject
                    for sub_token in child.subtree:
                        if sub_token != child:
                            for sub_token_idx, w_id in enumerate(word_ids):
                                if w_id is not None and doc[w_id] == sub_token:
                                    subject_indices.append(sub_token_idx)

    # If a subject-verb pair is found, create the attention link
    if subject_indices and verb_indices:
        for from_idx in subject_indices:
            for to_idx in verb_indices:
                predicted_matrix[from_idx, to_idx] = 1.0

    # Fallback and general rules
    # If no subject-verb pair is found, or for general sentence flow,
    # we can have a lighter attention from determiners and nouns to other nouns
    if not subject_indices or not verb_indices:
        for i in range(token_len):
            for j in range(token_len):
                if word_ids[i] is not None and word_ids[j] is not None:
                    from_token = doc[word_ids[i]]
                    to_token = doc[word_ids[j]]
                    if ("DET" in from_token.pos_ and "NOUN" in to_token.pos_) or \
                       ("NOUN" in from_token.pos_ and "NOUN" in to_token.pos_):
                        predicted_matrix[i, j] = 0.5
    
    # Add self-attention for all tokens to ensure a valid distribution
    for i in range(token_len):
        predicted_matrix[i, i] = 1.0

    # Normalize the matrix to represent a valid attention distribution (softmax-like)
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    normalized_matrix = predicted_matrix / row_sums

    return "Long-Distance Subject-Verb Alignment", normalized_matrix