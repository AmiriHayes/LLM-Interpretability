import numpy as np
import spacy

# Load the English language model from spaCy
# You may need to run: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    # A simplified mock NLP object to prevent crashing
    class MockDoc:
        def __init__(self, text):
            self.text = text
            self.ents = []
            self.tokens = [MockToken(t, 'NOUN') for t in text.split()]
    class MockToken:
        def __init__(self, text, pos):
            self.text = text
            self.pos_ = pos
            self.dep_ = ''
            self.head = self
            self.i = 0
    nlp = lambda x: MockDoc(x)


def parallel_elements_pattern(sentence, tokenizer):
    """
    Hypothesizes a 'Parallel Element Attender' pattern for a given sentence.
    This pattern predicts attention between words that are part of a coordinated
    list or phrase, sharing a common parent token and grammatical function.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer object with a `tokenize` method.

    Returns:
        tuple: A tuple containing the pattern name and the predicted attention matrix.
    """
    # Tokenize the sentence with the given BERT tokenizer
    tokens = tokenizer(sentence, return_tensors="pt")
    input_ids = tokens.input_ids[0]
    token_len = len(input_ids)
    word_ids = tokens.word_ids()

    # Initialize the attention matrix with zeros
    predicted_matrix = np.zeros((token_len, token_len))

    # Process the sentence with spaCy for dependency and POS information
    doc = nlp(sentence)

    # Dictionary to map sentence words to their BERT token indices
    word_to_token_map = {}
    for i, word in enumerate(doc):
        # find corresponding bert tokens for a word and add to map
        start_token_idx = -1
        for j, token_word_id in enumerate(word_ids):
            if token_word_id == i:
                if start_token_idx == -1:
                    start_token_idx = j
                word_to_token_map[word] = (start_token_idx, j)
    
    # Iterate through each token in the spaCy document
    for i, token in enumerate(doc):
        # Check for coordinating conjunctions (like 'and', 'or', 'but')
        # This is a key indicator of parallel structures
        if token.dep_ == "cc":
            # If a coordinator is found, find the head of the coordinated phrase
            head_token = token.head
            # Find all tokens that are children of the head and are also coordinated
            coordinated_children = [
                child for child in head_token.children 
                if child.dep_ in ["conj", "cc"]
            ]
            
            # Add the head token to the list of parallel elements
            parallel_elements = [head_token] + coordinated_children
            
            # If we found parallel elements
            if len(parallel_elements) > 1:
                # Get the BERT token indices for each element
                bert_indices = []
                for element in parallel_elements:
                    if element in word_to_token_map:
                        start, end = word_to_token_map[element]
                        bert_indices.append(list(range(start, end + 1)))
                
                # Flatten the list of indices
                flat_indices = [idx for sublist in bert_indices for idx in sublist]

                # Create attention links between all pairs of tokens in the parallel group
                for idx1 in flat_indices:
                    for idx2 in flat_indices:
                        if idx1 != idx2:
                            predicted_matrix[idx1, idx2] = 1.0
                            
    # Normalize the matrix rows so they sum to 1
    # This simulates the uniform attention distribution often seen in these patterns
    row_sums = predicted_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    predicted_matrix = predicted_matrix / row_sums[:, np.newaxis]

    return "Parallel Element Attender Pattern", predicted_matrix