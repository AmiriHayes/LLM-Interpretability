import numpy as np
import spacy

def punctuation_chaining_pattern(sentence: str, tokenizer) -> tuple[str, np.ndarray]:
    """
    Hypothesizes a 'Punctuation Chaining Pattern' attention model.

    This pattern is characterized by high attention from punctuation marks
    to the subsequent punctuation mark in the sentence, effectively forming
    a chain of dependencies along the sentence's structural boundaries.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer object (e.g., BertTokenizer).

    Returns:
        tuple[str, np.ndarray]: A tuple containing the pattern name and the
                                predicted attention matrix.
    """
    # Initialize the spacy nlp object once
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        
    doc = nlp(sentence)
    
    # Tokenize and get word IDs for aligning spacy tokens to BERT tokens
    tokens = tokenizer(sentence, return_tensors="np", add_special_tokens=True)
    word_ids = tokens.word_ids()
    seq_len = len(tokens["input_ids"][0])
    
    predicted_matrix = np.full((seq_len, seq_len), 0.1)
    
    punctuation_indices = []
    
    # Find indices of punctuation tokens
    for i, word_id in enumerate(word_ids):
        if word_id is not None:
            spacy_token = doc[word_id]
            if spacy_token.is_punct:
                punctuation_indices.append(i)
    
    # Create the punctuation chain pattern
    if punctuation_indices:
        # Link each punctuation mark to the next one in the sentence
        for i in range(len(punctuation_indices) - 1):
            from_idx = punctuation_indices[i]
            to_idx = punctuation_indices[i+1]
            predicted_matrix[from_idx, to_idx] = 0.8
            
        # Also link the first punctuation mark to the final sentence token
        predicted_matrix[punctuation_indices[0], seq_len-1] = 0.3
    
    # Every punctuation mark also attends to the final token
    for from_idx in punctuation_indices:
      predicted_matrix[from_idx, seq_len-1] = 0.3

    # Add self-attention for special tokens
    if seq_len > 0:
        predicted_matrix[0, 0] = 1.0  # [CLS] token
    if seq_len > 1:
        predicted_matrix[seq_len - 1, seq_len - 1] = 1.0  # [SEP] token
    
    # Normalize each row to ensure they sum to 1.0
    for i in range(seq_len):
        row_sum = np.sum(predicted_matrix[i, :])
        if row_sum > 0:
            predicted_matrix[i, :] /= row_sum
    
    return 'Punctuation Chaining Pattern', predicted_matrix

# Example usage:
# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# sentence = "The towering mountains, majestic and imposing, stood as silent sentinels, guarding the valley below, didn't they?"
# pattern_name, matrix = punctuation_chaining_pattern(sentence, tokenizer)
# print(pattern_name)
# print(matrix)