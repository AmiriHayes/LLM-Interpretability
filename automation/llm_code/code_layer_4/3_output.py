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

def parallel_structure_symmetry(sentence: str, tokenizer) -> tuple[str, np.ndarray]:
    """
    Hypothesizes a 'Symmetry in Parallel Structures' attention pattern.

    This pattern is characterized by high, often bidirectional, attention
    between tokens in parallel grammatical constructions like lists,
    appositive phrases, or repeated words.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer object (e.g., BertTokenizer).

    Returns:
        tuple[str, np.ndarray]: A tuple containing the pattern name and the
                                predicted attention matrix.
    """
    # Tokenize and get word IDs
    toks = tokenizer([sentence], return_tensors="np", add_special_tokens=True)
    input_ids = toks["input_ids"][0]
    word_ids = toks.word_ids()
    seq_len = len(input_ids)
    
    # Initialize a low-attention matrix
    predicted_matrix = np.full((seq_len, seq_len), 0.05)
    np.fill_diagonal(predicted_matrix, 0.1)

    # Use spaCy to analyze the sentence's linguistic structure
    doc = nlp(sentence)
    
    # Identify lists and parallel phrases based on punctuation and conjunctions
    list_items_indices = []
    
    for token in doc:
        # Check for list items
        if token.pos_ in ["NOUN", "VERB", "ADJ", "PROPN"] and (token.head.text == "," or token.head.text == "and" or token.head.text == "or"):
            list_items_indices.append(token.i)
        # Check for repeated phrases (simple case)
        for other_token in doc:
            if token.text.lower() == other_token.text.lower() and token.i != other_token.i:
                # Get the tokenizer indices for the tokens
                token_indices = [i for i, wid in enumerate(word_ids) if wid == token.i]
                other_token_indices = [i for i, wid in enumerate(word_ids) if wid == other_token.i]
                
                # Set strong bidirectional attention between them
                for i in token_indices:
                    for j in other_token_indices:
                        predicted_matrix[i, j] = 0.7
                        predicted_matrix[j, i] = 0.7
    
    # Get tokenizer indices for the list items
    token_indices = []
    for spacy_idx in list_items_indices:
        token_indices.extend([i for i, wid in enumerate(word_ids) if wid == spacy_idx])
    
    # Set high bidirectional attention between all identified list items
    for i in token_indices:
        for j in token_indices:
            if i != j:
                predicted_matrix[i, j] = 0.5
                
    # Normalize rows to sum to 1.0
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    predicted_matrix = np.divide(predicted_matrix, row_sums, out=np.zeros_like(predicted_matrix), where=row_sums!=0)

    return 'Symmetry in Parallel Structures', predicted_matrix

# Example usage:
# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# sentence = "The sun dipped below the horizon, painting the sky with vibrant hues of orange, pink, and purple."
# pattern_name, matrix = parallel_structure_symmetry(sentence, tokenizer)
# print(pattern_name)
# print(matrix)