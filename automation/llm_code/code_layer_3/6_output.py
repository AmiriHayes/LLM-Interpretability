import numpy as np
import spacy

# Load the spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def punctuation_and_list_aggregation(sentence, tokenizer):
    """
    Hypothesizes that this head aggregates attention around punctuation marks,
    particularly those that delimit lists, appositives, or parenthetical phrases.

    The function generates a predicted attention matrix by identifying
    punctuation and creating a pattern where tokens attend to the nearest
    significant punctuation mark (comma, colon, etc.) to their left. It also
    identifies words that are part of a list and makes them attend to each other
    and to the list's delimiter.

    Args:
        sentence (str): The input sentence.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.

    Returns:
        tuple: A tuple containing the pattern name and the predicted attention matrix.
    """
    # Tokenize the sentence and get the sequence length
    tokenized_sentence = tokenizer.tokenize(sentence)
    tokenized_sentence = ['[CLS]'] + tokenized_sentence + ['[SEP]']
    len_seq = len(tokenized_sentence)
    predicted_matrix = np.zeros((len_seq, len_seq), dtype=float)

    # Use spaCy to get linguistic information
    doc = nlp(sentence)
    
    # Find the indices of punctuation marks that act as list or clause delimiters
    punc_indices = []
    for i, token in enumerate(doc):
        # Using spaCy token.text to avoid issues with subwords
        # and match to the original sentence words
        if token.text in [',', ':', ';', "'", '"']:
            # Find the index of this token in the tokenized list
            token_idx = -1
            current_idx = 0
            for j, tok in enumerate(tokenized_sentence):
                if tok.startswith(token.text):
                    if j >= current_idx:
                        token_idx = j
                        break
            if token_idx != -1:
                punc_indices.append(token_idx)
    
    # All tokens attend to the nearest preceding significant punctuation mark
    # and to each other if they fall between the same two marks.
    for i in range(len_seq):
        # A token attends to itself
        predicted_matrix[i, i] += 1.0

        # Find the nearest punctuation mark to the left
        nearest_punc_idx = -1
        for punc_idx in reversed(punc_indices):
            if punc_idx < i:
                nearest_punc_idx = punc_idx
                break
        
        # If a punctuation mark is found, the current token attends to it
        if nearest_punc_idx != -1:
            predicted_matrix[i, nearest_punc_idx] += 1.0
        
        # All tokens also attend to the final punctuation mark of the sentence
        # The final token is often a period, question mark, or quotation mark.
        predicted_matrix[i, len_seq-1] += 1.0
        
        # The [CLS] token at index 0 should have self-attention and attention
        # to the final punctuation mark.
        predicted_matrix[0, 0] += 1.0
        predicted_matrix[0, len_seq-1] += 1.0

    # Punctuation marks also attend to other punctuation marks in the same series
    for i in range(len(punc_indices)):
        for j in range(len(punc_indices)):
            if i != j:
                predicted_matrix[punc_indices[i], punc_indices[j]] += 1.0

    # Normalize each row so that the sum of attention weights is 1
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero for any rows that might be all zeros
    predicted_matrix = np.divide(predicted_matrix, row_sums, out=np.zeros_like(predicted_matrix), where=row_sums != 0)

    return 'Punctuation and List Aggregation Pattern', predicted_matrix