import numpy as np
import spacy

# Load the spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy language model. This may take a moment.")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def coordinate_clause_alignment(sentence, tokenizer):
    """
    Predicts attention patterns for Layer 2, Head 7, based on the
    "Coordinate Clause Alignment" hypothesis.

    The function identifies coordinating conjunctions and commas that
    introduce or separate list items or clauses. It then assigns a
    high attention weight from these tokens to the first noun or verb
    in the coordinated phrase or list, simulating the observed
    attention pattern.

    Args:
        sentence (str): The input sentence.
        tokenizer: A Hugging Face tokenizer object.

    Returns:
        tuple: A tuple containing the pattern name and the predicted
               attention matrix.
    """
    toks = tokenizer([sentence], return_tensors="pt")
    input_ids = toks.input_ids[0]
    word_ids = toks.word_ids()
    len_seq = len(input_ids)
    
    # Initialize a zero matrix for attention
    predicted_matrix = np.zeros((len_seq, len_seq))

    # Parse the sentence with spaCy
    doc = nlp(sentence)
    
    # Store word indices to map spaCy tokens to BERT tokens
    word_to_bert_map = {}
    current_word_idx = -1
    for i, word_id in enumerate(word_ids):
        if word_id is not None and word_id != current_word_idx:
            current_word_idx = word_id
            word_to_bert_map[word_id] = i

    # Identify the heads of coordinate clauses and lists
    for token in doc:
        if token.dep_ == 'cc' or token.dep_ == 'punct':
            # Check for a conjunction or punctuation that coordinates items
            if token.head and token.head.pos_ in ['NOUN', 'VERB', 'ADJ']:
                # The head of the conjunction or punctuation is likely the
                # primary noun or verb in the coordinated list.
                head_token = token.head
                
                # Get the BERT token index for the head and the current token
                try:
                    head_idx = word_to_bert_map[head_token.i]
                    from_idx = word_to_bert_map[token.i]
                    
                    # For multi-token words, spaCy gives the start index. We might need to handle this.
                    # Simple approach: the first token of the word attends to the first token of its head.
                    
                    # Set a high attention weight from the coordinating token to the head
                    predicted_matrix[from_idx, head_idx] = 1.0
                    
                    # Additionally, give attention from other tokens in the coordinated phrase
                    # to the head of the list
                    
                    for child in head_token.children:
                        if child.i in word_to_bert_map and child.pos_ in ['NOUN', 'VERB', 'ADJ']:
                            child_idx = word_to_bert_map[child.i]
                            predicted_matrix[child_idx, head_idx] = 1.0

                except KeyError:
                    continue
    
    # Add self-attention for [CLS] and [SEP] tokens
    if len_seq > 0:
        predicted_matrix[0, 0] = 1.0
    if len_seq > 1:
        predicted_matrix[-1, -1] = 1.0
    
    # Normalize the matrix by row to ensure attention weights sum to 1
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    predicted_matrix = np.divide(predicted_matrix, row_sums, out=np.zeros_like(predicted_matrix), where=row_sums != 0)
    
    return 'Coordinate Clause Alignment', predicted_matrix