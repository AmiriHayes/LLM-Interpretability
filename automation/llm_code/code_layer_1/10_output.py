import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")

def co_referential_binding(sentence, tokenizer):
    """
    Hypothesizes a co-referential and associative binding pattern for Layer 1, Head 10.
    This pattern connects a head noun/pronoun to its descriptive elements,
    including adjectives, clauses, and lists, often mediated by punctuation
    or prepositions.
    
    The function generates a predicted attention matrix where a token (or its subword
    pieces) attends to its co-referential or associatively bound tokens.
    
    Parameters:
    - sentence (str): The input sentence.
    - tokenizer: The tokenizer object (e.g., from the Hugging Face library).
    
    Returns:
    - tuple: A string with the pattern name and a NumPy array for the predicted
             attention matrix.
    """
    
    # Use Spacy to parse the sentence for linguistic information
    doc = nlp(sentence)
    
    # Tokenize the sentence with the given tokenizer
    tokenized = tokenizer([sentence], return_tensors="pt")
    input_ids = tokenized.input_ids[0]
    word_ids = tokenized.word_ids(batch_index=0)
    
    # Get the sequence length and initialize the matrix
    len_seq = len(input_ids)
    predicted_matrix = np.zeros((len_seq, len_seq))
    
    # Create a mapping from Spacy tokens to BERT subword tokens
    spacy_to_bert_map = {}
    for i, word_idx in enumerate(word_ids):
        if word_idx is not None:
            if word_idx not in spacy_to_bert_map:
                spacy_to_bert_map[word_idx] = []
            spacy_to_bert_map[word_idx].append(i)

    # Core logic for the co-referential binding pattern
    for spacy_token in doc:
        head_token_idx = spacy_token.i
        # Find BERT token indices for the current Spacy token
        head_bert_indices = spacy_to_bert_map.get(head_token_idx, [])
        
        # Look for descriptive adjectives, appositive phrases, or list items
        # Spacy's dependency parser helps identify these relationships
        for child in spacy_token.children:
            child_token_idx = child.i
            child_bert_indices = spacy_to_bert_map.get(child_token_idx, [])

            # Check if the child is a descriptive modifier
            # The 'amod' (adjectival modifier), 'appos' (appositive),
            # and 'pobj' (object of a preposition) relationships are key.
            if child.dep_ in ["amod", "appos", "pobj", "cc"]:
                for from_idx in child_bert_indices:
                    for to_idx in head_bert_indices:
                        predicted_matrix[from_idx, to_idx] = 1.0
            
            # Also, look for relationships where the head is the modifier of the child
            if spacy_token.dep_ in ["amod", "appos"]:
                for from_idx in head_bert_indices:
                    for to_idx in child_bert_indices:
                        predicted_matrix[from_idx, to_idx] = 1.0

        # Account for lists and conjunctions
        if spacy_token.dep_ in ["conj"] and spacy_token.head:
            head_of_conj_bert_indices = spacy_to_bert_map.get(spacy_token.head.i, [])
            conj_bert_indices = spacy_to_bert_map.get(spacy_token.i, [])
            
            for from_idx in conj_bert_indices:
                for to_idx in head_of_conj_bert_indices:
                    predicted_matrix[from_idx, to_idx] = 1.0

    # Punctuation handling: Punctuation often attends to the word it follows
    # or the words that it separates. This part of the pattern is more heuristic.
    for i, word_id in enumerate(word_ids):
        if word_id is None: # Punctuation or special tokens
            if input_ids[i] == tokenizer.sep_token_id: # End-of-sentence token
                # This token can attend to the last word and the CLS token
                predicted_matrix[i, -2] = 1.0
                predicted_matrix[i, 0] = 1.0
            elif input_ids[i] == tokenizer.cls_token_id: # CLS token
                # CLS token often attends to itself
                predicted_matrix[i, i] = 1.0
            
            # For commas, colons, etc., connect them to the word they follow
            if i > 0 and word_ids[i-1] is not None:
                prev_word_end_idx = i - 1
                while prev_word_end_idx > 0 and word_ids[prev_word_end_idx] is None:
                    prev_word_end_idx -= 1
                
                if word_ids[prev_word_end_idx] is not None:
                    prev_word_spacy_idx = word_ids[prev_word_end_idx]
                    
                    # Connect punctuation to the previous word and its referent
                    prev_word_bert_indices = spacy_to_bert_map.get(prev_word_spacy_idx, [])
                    if prev_word_bert_indices:
                        for to_idx in prev_word_bert_indices:
                             predicted_matrix[i, to_idx] = 1.0


    # Normalize the matrix by row to simulate uniform attention from each token
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    normalized_matrix = predicted_matrix / row_sums
    
    return 'Co-referential and Associative Binding Pattern', normalized_matrix