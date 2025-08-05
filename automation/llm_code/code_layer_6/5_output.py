import numpy as np
import spacy
from typing import Tuple
from transformers import AutoTokenizer

def adjectival_noun_alignment(sentence: str, tokenizer: AutoTokenizer) -> Tuple[str, np.ndarray]:
    """
    Hypothesizes the attention pattern for Layer 6, Head 5.
    This function identifies noun phrases and assigns attention from pre-nominal
    modifiers (determiners, adjectives) to their head noun. It also
    handles attention between multi-token words and their parts.
    
    Args:
        sentence (str): The input sentence string.
        tokenizer: The tokenizer to use (e.g., AutoTokenizer).
    
    Returns:
        Tuple[str, np.ndarray]: A tuple containing the pattern name and the
                                predicted attention matrix of size (token_len, token_len).
    """
    # Load a spaCy model for linguistic analysis
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise OSError("spaCy model 'en_core_web_sm' not found. "
                      "Please install with: python -m spacy download en_core_web_sm")

    doc = nlp(sentence)
    
    # Tokenize the sentence with the given tokenizer
    encoded_input = tokenizer(
        sentence,
        return_tensors="pt",
        add_special_tokens=True,
        return_offsets_mapping=True
    )
    word_ids = encoded_input.word_ids(batch_index=0)
    token_len = len(encoded_input.input_ids[0])
    
    predicted_matrix = np.zeros((token_len, token_len))

    # Helper function to get all token indices for a given spaCy token's word_id
    def get_token_indices_for_spacy_token(spacy_token, word_ids):
        # find the start index of the spacy token in the word_ids array
        # this is necessary because word_ids is a list of integers
        # where consecutive integers represent sub-tokens of the same word
        start_word_id = spacy_token.i
        end_word_id = spacy_token.i

        token_indices = [i for i, x in enumerate(word_ids) if x == start_word_id]
        
        # If the word is a multi-token word, find the rest of its tokens
        if len(token_indices) == 0:
            return [] # This token was not mapped, skip
            
        current_word_id = word_ids[token_indices[-1]]
        next_token_index = token_indices[-1] + 1
        
        while next_token_index < len(word_ids) and word_ids[next_token_index] == current_word_id:
            token_indices.append(next_token_index)
            next_token_index += 1

        return token_indices

    # Iterate through spaCy tokens to find noun chunks and modifiers
    for chunk in doc.noun_chunks:
        head_token = chunk.root
        head_token_indices = get_token_indices_for_spacy_token(head_token, word_ids)
        
        if not head_token_indices:
            continue

        # Look for pre-nominal modifiers within the noun chunk
        for token in chunk:
            # Check if the token is a modifier and precedes the head
            if (token.dep_ in ["det", "amod", "poss"] or token.pos_ in ["DET", "ADJ", "PRON"]) and token.i < head_token.i:
                modifier_indices = get_token_indices_for_spacy_token(token, word_ids)
                
                if not modifier_indices:
                    continue
                
                # Assign attention from each modifier token to all head tokens
                for mod_idx in modifier_indices:
                    for head_idx in head_token_indices:
                        predicted_matrix[mod_idx, head_idx] = 1.0

    # Handle multi-token words (e.g., 'sub-tokens')
    current_word_id = -1
    sub_token_indices = []
    
    for i, w_id in enumerate(word_ids):
        if w_id is not None:
            if w_id != current_word_id and current_word_id != -1:
                if len(sub_token_indices) > 1:
                    for from_idx in sub_token_indices:
                        for to_idx in sub_token_indices:
                            predicted_matrix[from_idx, to_idx] = 1.0
                sub_token_indices = []
            
            sub_token_indices.append(i)
            current_word_id = w_id

    if len(sub_token_indices) > 1:
        for from_idx in sub_token_indices:
            for to_idx in sub_token_indices:
                predicted_matrix[from_idx, to_idx] = 1.0

    # Ensure special tokens have attention, mainly self-attention
    for i, w_id in enumerate(word_ids):
        if w_id is None:
            predicted_matrix[i, i] = 1.0

    # Normalize the matrix to represent a valid attention distribution
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    normalized_matrix = predicted_matrix / row_sums

    return "Adjectival and Noun Phrase Alignment Pattern", normalized_matrix