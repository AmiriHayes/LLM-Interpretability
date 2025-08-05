import numpy as np
import spacy

def parenthesis_pattern(sentence: str, tokenizer) -> tuple[str, np.ndarray]:
    """
    Predicts the attention pattern for Layer 3, Head 0, which is responsible
    for the 'Parenthesis Pattern'.

    This function identifies pairs of punctuation marks that enclose clauses or phrases
    (e.g., '...', "...", (...) and ,...,,) and creates a matrix where these paired
    tokens have high bidirectional attention. It also gives attention from tokens
    within the enclosed phrase to the surrounding punctuation and other tokens.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer object (e.g., from Hugging Face).

    Returns:
        tuple[str, np.ndarray]: A tuple containing the pattern name and the
                                predicted attention matrix.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    
    tokens = tokenizer([sentence], return_tensors="pt")
    token_ids = tokens.input_ids[0].tolist()
    token_len = len(token_ids)
    
    # Initialize the attention matrix with zeros
    predicted_matrix = np.zeros((token_len, token_len), dtype=float)

    # Use tokenizer's word_ids to map spaCy tokens to BERT tokens
    word_ids = tokens.word_ids(batch_index=0)
    
    # Map spacy tokens to token_ids for BERT
    spacy_to_bert_map = {
        spacy_token_idx: [i for i, bert_word_id in enumerate(word_ids) if bert_word_id == spacy_token_idx]
        for spacy_token_idx, _ in enumerate(doc)
    }

    # Identify pairs of enclosing punctuation based on spacy's token info
    punctuation_pairs = []
    stack = []
    
    for i, token in enumerate(doc):
        # Handle commas, quotes, and other parenthesis-like structures
        if token.text in ['"', "'", '(', '[', '{'] or token.text == ',':
            stack.append((token.text, i))
        elif (token.text in ['"', "'", ')', ']', '}'] or token.text == ',') and stack:
            # Pop the last opening punctuation
            opening_text, opening_idx = stack.pop()
            
            # Check for matching pairs (e.g., '(' with ')') or comma pairs
            is_match = False
            if (opening_text == '(' and token.text == ')') or \
               (opening_text == '[' and token.text == ']') or \
               (opening_text == '{' and token.text == '}') or \
               (opening_text == "'" and token.text == "'") or \
               (opening_text == '"' and token.text == '"') or \
               (opening_text == ',' and token.text == ','):
                is_match = True

            if is_match:
                punctuation_pairs.append((opening_idx, i))
                
    # Loop through the identified pairs and create the attention pattern
    for start_idx, end_idx in punctuation_pairs:
        # Get the BERT token indices for the opening and closing punctuation
        start_bert_indices = spacy_to_bert_map.get(start_idx, [])
        end_bert_indices = spacy_to_bert_map.get(end_idx, [])

        # Assign high bidirectional attention between paired punctuation
        for i in start_bert_indices:
            for j in end_bert_indices:
                predicted_matrix[i, j] = 1.0
                predicted_matrix[j, i] = 1.0
        
        # Assign some attention from tokens inside the pair to the boundary punctuation
        for bert_idx in range(start_bert_indices[-1] + 1, end_bert_indices[0]):
            for punc_bert_idx in start_bert_indices + end_bert_indices:
                predicted_matrix[bert_idx, punc_bert_idx] = 0.5
    
    # Assign self-attention to the CLS token and a placeholder for the EOS token
    predicted_matrix[0, 0] = 1.0
    predicted_matrix[token_len - 1, 0] = 1.0

    # Normalize the matrix rows so they sum to 1
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero
    predicted_matrix = np.divide(predicted_matrix, row_sums, out=np.zeros_like(predicted_matrix), where=row_sums != 0)

    return 'Parenthesis Pattern', predicted_matrix