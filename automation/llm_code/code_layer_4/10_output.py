import numpy as np
import spacy
from typing import Tuple

def subject_verb_linking(sentence: str, tokenizer) -> Tuple[str, np.ndarray]:
    """
    Hypothesizes a 'Subject-Verb Linking' attention model.

    This pattern identifies and assigns attention from verbs, participles, or
    modifiers to the main subject of their respective clauses. It's a fundamental
    component of syntactic and semantic parsing.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer object (e.g., BertTokenizer).

    Returns:
        tuple[str, np.ndarray]: A tuple containing the pattern name and the
                                predicted attention matrix.
    """
    # Load the spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model 'en_core_web_sm'...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    # Tokenize the sentence with BERT tokenizer to get word_ids
    tokens = tokenizer([sentence], return_tensors="np", max_length=512, truncation=True)
    word_ids = tokens.word_ids(batch_index=0)
    seq_len = len(tokens['input_ids'][0])

    # Initialize the attention matrix with a low base value
    predicted_matrix = np.full((seq_len, seq_len), 0.1)

    # Process the sentence with spaCy
    doc = nlp(sentence)

    # A helper function to find the BERT token indices for a given spaCy token
    def get_token_indices(spacy_token, word_ids):
        indices = []
        for i, word_id in enumerate(word_ids):
            if word_id == spacy_token.i:
                indices.append(i)
        return indices

    for token in doc:
        # Find subjects (nsubj, csubj, etc.)
        if token.dep_ in ["nsubj", "csubj", "nsubjpass", "agent"]:
            subject_token = token
            # The head of the subject is the verb
            verb_token = token.head
            
            # Find modifiers of the verb (adverbs, participles, etc.)
            verb_modifiers = [
                child for child in verb_token.children
                if child.pos_ in ["ADJ", "VERB", "ADV"] or child.dep_ in ["advcl", "dobj"]
            ]
            
            # Get BERT indices for the subject and verb
            subject_indices = get_token_indices(subject_token, word_ids)
            verb_indices = get_token_indices(verb_token, word_ids)
            
            # Assign strong attention from the verb to the subject
            for v_idx in verb_indices:
                for s_idx in subject_indices:
                    predicted_matrix[v_idx, s_idx] = 0.8
            
            # Assign attention from verb modifiers to the subject
            for modifier in verb_modifiers:
                modifier_indices = get_token_indices(modifier, word_ids)
                for mod_idx in modifier_indices:
                    for s_idx in subject_indices:
                        predicted_matrix[mod_idx, s_idx] = 0.5
        
        # Handle cases where a verb modifies the subject directly, like in "The sun, painting the sky, dipped..."
        if token.dep_ in ["acl", "advcl", "amod"]:
            modified_noun = token.head
            modifier_indices = get_token_indices(token, word_ids)
            noun_indices = get_token_indices(modified_noun, word_ids)
            
            for mod_idx in modifier_indices:
                for n_idx in noun_indices:
                    predicted_matrix[mod_idx, n_idx] = 0.6

    # Add strong self-attention for special tokens
    if seq_len > 0:
        predicted_matrix[0, 0] = 1.0  # [CLS] token
    if seq_len > 1:
        predicted_matrix[-1, -1] = 1.0 # [SEP] token
        
    # Normalize each row to sum to 1.0
    for i in range(seq_len):
        row_sum = np.sum(predicted_matrix[i, :])
        if row_sum > 0:
            predicted_matrix[i, :] /= row_sum

    return 'Subject-Verb Linking Pattern', predicted_matrix

# Example usage:
# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# sentence = "The sun dipped below the horizon, painting the sky with vibrant hues of orange, pink, and purple."
# pattern_name, matrix = subject_verb_linking(sentence, tokenizer)
# print(pattern_name)
# print(matrix)