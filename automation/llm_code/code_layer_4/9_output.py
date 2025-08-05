import numpy as np
import spacy
from typing import Tuple

# Load the spacy model once
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'.")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def appositive_clause_linking(sentence: str, tokenizer) -> Tuple[str, np.ndarray]:
    """
    Hypothesizes an 'Appositive and Clause Boundary Linking' attention model.

    This pattern assumes attention flows from the heads of dependent clauses
    (appositives, gerunds, etc.) and their boundary commas back to the head
    of the main clause they modify.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer object (e.g., BertTokenizer).

    Returns:
        tuple[str, np.ndarray]: A tuple containing the pattern name and the
                                predicted attention matrix.
    """
    # Tokenize the sentence and get the sequence length
    tokens = tokenizer([sentence], return_tensors="np", max_length=512, truncation=True)
    input_ids = tokens['input_ids'][0]
    seq_len = len(input_ids)
    
    # Initialize the predicted attention matrix with zeros
    predicted_matrix = np.zeros((seq_len, seq_len))

    # Use SpaCy to parse the sentence
    doc = nlp(sentence)
    
    # Create a mapping from SpaCy token index to BERT token indices
    word_ids = tokens.word_ids(batch_index=0)
    word_to_token_map = {}
    for i, word_idx in enumerate(word_ids):
        if word_idx is not None:
            if word_idx not in word_to_token_map:
                word_to_token_map[word_idx] = []
            word_to_token_map[word_idx].append(i)

    # Find the root of the sentence (the main verb)
    main_verb_idx = -1
    for token in doc:
        if token.dep_ == 'ROOT':
            if token.i in word_to_token_map:
                main_verb_idx = token.i
            break

    if main_verb_idx != -1:
        # Loop through each token in the SpaCy doc
        for token in doc:
            # Check for clauses or appositives, often signaled by commas
            # and a descriptive phrase
            is_clause_boundary = token.text == ','
            is_gerund = token.pos_ == 'VERB' and token.tag_ == 'VBG'
            is_appositive = token.dep_ == 'appos'
            
            # Identify the head of the current phrase
            head_idx = token.head.i if token.head else -1
            
            # If the current token is a comma, gerund, or appositive, and it
            # has a head, link it to the main verb or subject.
            if (is_clause_boundary or is_gerund or is_appositive) and head_idx != -1:
                # Find the token indices for the current token and its head
                if token.i in word_to_token_map and main_verb_idx in word_to_token_map:
                    from_token_indices = word_to_token_map[token.i]
                    to_token_indices = word_to_token_map[main_verb_idx]
                    
                    # Distribute attention from all subtokens of 'from' to all of 'to'
                    for from_idx in from_token_indices:
                        for to_idx in to_token_indices:
                            predicted_matrix[from_idx, to_idx] = 1.0

            # Special case: Link punctuation marks at the end of a clause
            # back to the start of the clause
            if token.text in [',', '!', '?', ':'] and head_idx != -1:
                 # Check if the punctuation is a clause boundary, not just a list
                 if token.dep_ in ['punct'] and token.i in word_to_token_map:
                     from_token_indices = word_to_token_map[token.i]
                     # The target is the word that this punctuation modifies
                     to_token_indices = word_to_token_map.get(token.head.i, [])
                     if not to_token_indices and doc[0].i in word_to_token_map:
                         # Fallback to linking to the start of the sentence
                         to_token_indices = word_to_token_map[doc[0].i]
                     
                     for from_idx in from_token_indices:
                         for to_idx in to_token_indices:
                             predicted_matrix[from_idx, to_idx] = 1.0

    # Ensure self-attention for special tokens
    if seq_len > 0:
        predicted_matrix[0, 0] = 1.0  # [CLS]
    if seq_len > 1:
        predicted_matrix[-1, -1] = 1.0  # [SEP]
        
    # Normalize each row to sum to 1.0 to get a probability distribution
    for i in range(seq_len):
        row_sum = np.sum(predicted_matrix[i, :])
        if row_sum > 0:
            predicted_matrix[i, :] /= row_sum

    return 'Appositive and Clause Boundary Linking Pattern', predicted_matrix