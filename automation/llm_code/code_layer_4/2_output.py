import numpy as np
import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def punctuation_bridging_pattern(sentence: str, tokenizer) -> tuple[str, np.ndarray]:
    """
    Hypothesizes a 'Punctuation Bridging Pattern' attention pattern.

    This pattern is characterized by high, uniform attention between
    all punctuation tokens in a sentence.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer object (e.g., BertTokenizer).

    Returns:
        tuple[str, np.ndarray]: A tuple containing the pattern name and the
                                predicted attention matrix.
    """
    toks = tokenizer([sentence], return_tensors="np")
    input_ids = toks["input_ids"][0]
    seq_len = len(input_ids)
    
    # Initialize a matrix with a low default attention value for all tokens.
    predicted_matrix = np.full((seq_len, seq_len), 0.05)

    # Use spaCy to identify punctuation tokens
    doc = nlp(sentence)
    word_ids = toks.word_ids()[1:-1] # Exclude special tokens
    
    punctuation_indices = []
    current_token_index = 0
    for i, word_id in enumerate(word_ids):
        if doc[word_id].is_punct:
            punctuation_indices.append(i + 1) # +1 for [CLS] token

    # For each punctuation token, have it attend to all other punctuation tokens
    # with a high weight.
    if punctuation_indices:
        for i in punctuation_indices:
            for j in punctuation_indices:
                predicted_matrix[i, j] = 0.5
        
    # Ensure some self-attention for all tokens, including punctuation.
    np.fill_diagonal(predicted_matrix, predicted_matrix.diagonal() + 0.1)

    # Normalize each row to sum to 1.0.
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    predicted_matrix = np.divide(predicted_matrix, row_sums, out=np.zeros_like(predicted_matrix), where=row_sums!=0)

    return 'Punctuation Bridging Pattern', predicted_matrix

# Example usage:
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# sentence = "The old, creaky house, standing on the hill, seemed to whisper secrets."
# pattern_name, matrix = punctuation_bridging_pattern(sentence, tokenizer)
# print(pattern_name)
# print(matrix)