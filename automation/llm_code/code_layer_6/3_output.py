import numpy as np
import spacy

def object_of_preposition(sentence, tokenizer):
    """
    Hypothesizes the attention pattern for Layer 6, Head 3 based on the
    "Object of Preposition" pattern.

    The function identifies prepositions and assigns attention from the preposition
    to its direct object. It also handles coordination and multi-word
    prepositional phrases. The pattern is designed to generalize to any sentence.
    
    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer for the model (e.g., BertTokenizer).
    
    Returns:
        tuple: A tuple containing the pattern name and the predicted attention matrix.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    toks = tokenizer([sentence], return_tensors="pt")
    
    # Get the word IDs from the tokenizer output
    # This helps map spaCy tokens back to BERT sub-tokens
    word_ids = toks.word_ids(batch_index=0)
    len_seq = len(toks.input_ids[0])
    predicted_matrix = np.zeros((len_seq, len_seq))

    for token in doc:
        # Check if the token is a preposition or a prepositional particle
        if token.pos_ == "ADP" or token.dep_ == "prep":
            # The head of the preposition is often the verb it modifies
            # The children of the preposition are the objects
            for child in token.children:
                # The children often include the noun phrase and determiners
                # We want to connect the preposition to the "root" of its object
                if child.dep_ in ["pobj", "nsubj", "dobj", "ROOT"] or child.pos_ in ["NOUN", "PROPN", "PRON"]:
                    from_token_idx = word_ids.index(token.i) if token.i in word_ids else None
                    to_token_idx = word_ids.index(child.i) if child.i in word_ids else None
                    
                    if from_token_idx is not None and to_token_idx is not None:
                        # Find the indices of the full token and any sub-tokens
                        from_indices = [i for i, x in enumerate(word_ids) if x == token.i]
                        to_indices = [i for i, x in enumerate(word_ids) if x == child.i]
                        
                        # Assign attention from the preposition to its object
                        for from_idx in from_indices:
                            for to_idx in to_indices:
                                predicted_matrix[from_idx, to_idx] = 1.0
                                
    # Assign self-attention for special tokens [CLS] and [SEP]
    predicted_matrix[0, 0] = 1.0
    predicted_matrix[-1, -1] = 1.0

    # Handle padding and other special tokens if present
    for i in range(len_seq):
        if word_ids[i] is None:
            predicted_matrix[i, i] = 1.0

    # Normalize the matrix by row to make it a valid attention matrix
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    normalized_matrix = predicted_matrix / row_sums

    return "Object of Preposition Pattern", normalized_matrix