import numpy as np
import spacy

# Load the spaCy model. This might need to be downloaded first
# with: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def enumeration_pattern(sentence, tokenizer):
    """
    Hypothesizes that this head identifies and links tokens within a list or series,
    often marked by commas and conjunctions.

    The function first identifies the items in any enumeration (list) using spaCy's
    dependency parsing. It then creates a predicted attention matrix where a token
    attends to the previous item in the list or the punctuation separating them.
    This includes attention from 'and' to the preceding list item, and from a comma
    to the preceding comma or list item.

    Args:
        sentence (str): The input sentence.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.

    Returns:
        tuple: A tuple containing the pattern name and the predicted attention matrix.
    """
    doc = nlp(sentence)
    tokens = tokenizer(sentence, return_tensors="pt")
    len_seq = len(tokens.input_ids[0])
    
    # Initialize predicted matrix with self-attention for all tokens,
    # including [CLS] and [SEP].
    predicted_matrix = np.eye(len_seq, dtype=float)

    # Use spaCy to get the token-to-word alignment for the BERT tokenizer
    token_word_ids = tokens.word_ids()

    # Find the indices of enumeration items and conjunctions
    list_items = []
    conjunction_indices = []

    # Iterate through spaCy tokens to find lists
    for i, token in enumerate(doc):
        # A simple heuristic for lists: items with 'conj' or 'cc' dependency
        if token.dep_ == "conj" or token.dep_ == "cc":
            # Find the BERT token indices for this spaCy token
            bert_indices = [idx for idx, word_id in enumerate(token_word_ids) if word_id == i]
            if bert_indices:
                if token.dep_ == "cc":
                    conjunction_indices.append(bert_indices[0])
                else:
                    list_items.append(bert_indices[0])
    
    # If there's a list, build the attention pattern
    if list_items or conjunction_indices:
        # Combine all relevant indices and sort them
        all_list_indices = sorted(list_items + conjunction_indices)

        # Create the attention chain
        for i in range(len(all_list_indices)):
            from_idx = all_list_indices[i]
            if i > 0:
                to_idx = all_list_indices[i-1]
                # High attention from a list item or conjunction to the previous one
                predicted_matrix[from_idx, to_idx] = 1.0
                
            # Additional attention from the list head to its first item
            if i == 0 and len(list_items) > 1:
                list_head_spacy_idx = doc[list_items[0]].head.i
                # Find the BERT token index for the list head
                list_head_bert_indices = [idx for idx, word_id in enumerate(token_word_ids) if word_id == list_head_spacy_idx]
                if list_head_bert_indices:
                    predicted_matrix[from_idx, list_head_bert_indices[0]] = 0.5
    
    # Normalize rows to sum to 1
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    predicted_matrix = np.divide(predicted_matrix, row_sums, out=np.zeros_like(predicted_matrix), where=row_sums != 0)

    return 'Enumeration Pattern', predicted_matrix