import numpy as np
import spacy
from spacy.tokens import Doc

# A simple list of common gerund/participle endings for a heuristic check
participle_endings = ['ing', 'ed']

def cross_clause_thematic_linking(sentence, tokenizer):
    """
    Predicts the attention matrix for Layer 3, Head 1 based on the Cross-Clause Thematic Linking pattern.

    This head is hypothesized to connect the main subject/verb of a sentence
    to thematically related verbs and nouns in subordinate or descriptive clauses.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer for the model (e.g., BertTokenizer).

    Returns:
        tuple: A tuple containing the pattern name and the predicted attention matrix.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)

    tokens = tokenizer([sentence], return_tensors="pt")
    input_ids = tokens.input_ids[0].tolist()
    token_len = len(input_ids)
    word_ids = tokens.word_ids()

    predicted_matrix = np.zeros((token_len, token_len))

    # Add self-attention for all tokens
    np.fill_diagonal(predicted_matrix, 1.0)

    # Find the main subject and verb of the sentence
    main_verb_indices = []
    main_subject_indices = []
    
    # Heuristically find the main verb and its subject.
    # This is a simplification; a full dependency parser would be more robust.
    for i, token in enumerate(doc):
        # We look for a root token to find the main verb
        if token.dep_ == 'ROOT':
            for j in range(len(word_ids)):
                if word_ids[j] == i:
                    main_verb_indices.append(j)
            
            # Find the subject (nsubj) of the root verb
            for child in token.children:
                if child.dep_ == 'nsubj':
                    for k in range(len(word_ids)):
                        if word_ids[k] == child.i:
                            main_subject_indices.append(k)

    # Find thematically related tokens in other clauses
    thematic_tokens = []
    for i, token in enumerate(doc):
        # Check for participles, gerunds, and related nouns that are not in the main clause
        if token.dep_ in ['advcl', 'acl', 'pobj', 'dobj'] or token.pos_ in ['VERB', 'NOUN']:
            # Ensure the token is not part of the main verb/subject
            is_main_verb = any(word_ids[j] == i for j in main_verb_indices)
            is_main_subject = any(word_ids[j] == i for j in main_subject_indices)
            
            if not (is_main_verb or is_main_subject):
                for k in range(len(word_ids)):
                    if word_ids[k] == i:
                        thematic_tokens.append(k)

    # Assign high attention from thematic tokens to the main verb and subject
    # and vice-versa. This models the cross-clause linking.
    for thematic_idx in thematic_tokens:
        for verb_idx in main_verb_indices:
            predicted_matrix[thematic_idx, verb_idx] += 1
            predicted_matrix[verb_idx, thematic_idx] += 1
        for subject_idx in main_subject_indices:
            predicted_matrix[thematic_idx, subject_idx] += 1
            predicted_matrix[subject_idx, thematic_idx] += 1

    # Normalize each row to sum to 1 to produce a valid attention matrix
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero
    predicted_matrix = np.where(row_sums == 0, 0, predicted_matrix / row_sums)

    return 'Cross-Clause Thematic Linking', predicted_matrix