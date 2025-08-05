import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")

def list_delimitation_pattern(sentence, tokenizer):
    """
    Hypothesizes the attention pattern for Layer 5, Head 0 as a 'List Delimitation Pattern'.

    This function predicts that the head attends from a list item (token) to the
    punctuation or conjunction (delimiter) that precedes it. It uses spacy to
    identify list items and their delimiters within the sentence.

    Parameters:
    - sentence (str): The input sentence.
    - tokenizer: A BERT tokenizer instance.

    Returns:
    - tuple: A tuple containing the name of the pattern and the predicted
             attention matrix.
    """
    toks = tokenizer([sentence], return_tensors="pt")
    input_ids = toks.input_ids[0]
    word_ids = toks.word_ids(batch_index=0)
    len_seq = len(input_ids)
    
    out = np.zeros((len_seq, len_seq))

    doc = nlp(sentence)
    
    # Create a mapping from spacy token index to BERT token index
    spacy_to_bert = {}
    bert_to_spacy = {}
    bert_idx = 1 # Start after [CLS]
    
    for i, spacy_token in enumerate(doc):
        # Handle subword tokens from BERT tokenizer
        for j in range(bert_idx, len_seq - 1):
            if word_ids[j] == i:
                if i not in spacy_to_bert:
                    spacy_to_bert[i] = []
                spacy_to_bert[i].append(j)
                bert_to_spacy[j] = i
                bert_idx = j + 1
            else:
                break

    for i, spacy_token in enumerate(doc):
        # Identify list items and their delimiters using spacy's part-of-speech tags and dependency parsing
        # The pattern is: list item (NOUN, PROPN, VERB, ADJ) followed by a punctuation or conjunction
        if spacy_token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"] and i > 0:
            # Check for preceding comma, conjunction (and, or), or other punctuation
            prev_token = doc[i-1]
            if prev_token.text in [",", "and", "or", "but"]:
                
                # Get BERT indices for the current token and the previous token
                if i in spacy_to_bert and (i-1) in spacy_to_bert:
                    from_indices = spacy_to_bert[i]
                    to_indices = spacy_to_bert[i-1]
                    
                    # For each subword of the list item, attend to each subword of the delimiter
                    for from_idx in from_indices:
                        for to_idx in to_indices:
                            out[from_idx, to_idx] = 1.0

    # Handle special tokens
    out[0, 0] = 1.0  # [CLS] self-attention
    out[-1, -1] = 1.0 # [SEP] self-attention

    # Normalize the matrix by row to represent uniform attention distribution
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 # Avoid division by zero
    predicted_matrix = out / row_sums

    return 'List Delimitation Pattern', predicted_matrix