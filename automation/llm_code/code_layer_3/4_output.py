import numpy as np
import spacy
from transformers import BertTokenizer

# Load spaCy's English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def parallel_structure_pattern(sentence, tokenizer):
    """
    Predicts attention patterns for a head responsible for recognizing and
    linking elements within parallel structures or lists.

    Args:
        sentence (str): The input sentence.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.

    Returns:
        tuple: A tuple containing the pattern name and the predicted attention matrix.
    """
    # Tokenize the sentence and get token IDs
    tokens = tokenizer([sentence], return_tensors="pt")
    len_seq = len(tokens.input_ids[0])
    
    # Initialize the attention matrix with zeros
    predicted_matrix = np.zeros((len_seq, len_seq), dtype=float)

    # Use spaCy to parse the sentence
    doc = nlp(sentence)
    
    # Get word IDs from the tokenizer to align with spaCy tokens
    word_ids = tokens.word_ids(batch_index=0)
    
    # Map spaCy tokens to BERT token indices
    spacy_to_bert_map = {}
    for i, word_idx in enumerate(word_ids):
        if word_idx is not None:
            if word_idx not in spacy_to_bert_map:
                spacy_to_bert_map[word_idx] = []
            spacy_to_bert_map[word_idx].append(i)

    # Identify parallel structures using spaCy's dependency parser and part-of-speech tags
    # This rule looks for tokens with 'conj' (conjunction) dependency, which are
    # key indicators of parallel structures. It then draws attention from the
    # head of the conjunction to the conjoined elements.
    for token in doc:
        # Check if the token is part of a conjunction
        if token.dep_ == "conj":
            # The 'head' of a conjunction is the main item in the list/structure
            head_token = token.head
            
            # Map spaCy tokens to BERT token indices
            head_indices = spacy_to_bert_map.get(head_token.i - 1, [])
            conj_indices = spacy_to_bert_map.get(token.i - 1, [])

            # Draw attention from the head to all conjoined tokens
            for h_idx in head_indices:
                for c_idx in conj_indices:
                    predicted_matrix[c_idx, h_idx] = 1.0  # Attention from conjoined item to the head

            # Also, draw attention from the head to all conjoined tokens
            for h_idx in head_indices:
                for c_idx in conj_indices:
                    predicted_matrix[h_idx, c_idx] = 1.0  # Attention from head to conjoined item

    # Add self-attention for all tokens
    for i in range(len_seq):
        predicted_matrix[i, i] = 1.0

    # Assign special attention for [CLS] and [SEP] tokens
    # [CLS] token attends to the first word (often the subject or a key noun)
    cls_idx = spacy_to_bert_map.get(0, [1])[0] # Default to index 1 if not found
    predicted_matrix[cls_idx, 0] = 1.0 
    predicted_matrix[0, cls_idx] = 1.0
    
    # Normalize each row to ensure all values in a row sum to 1.
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero for empty rows
    predicted_matrix = np.divide(predicted_matrix, row_sums, out=np.zeros_like(predicted_matrix), where=row_sums!=0)

    return 'Parallel Structure Pattern', predicted_matrix