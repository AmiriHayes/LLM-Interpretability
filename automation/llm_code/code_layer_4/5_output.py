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

def adjectival_and_noun_phrase_binding(sentence: str, tokenizer) -> tuple[str, np.ndarray]:
    """
    Hypothesizes an 'Adjectival and Noun Phrase Binding' attention pattern.

    This pattern is characterized by high attention from modifiers (adjectives,
    determiners, etc.) to the head of the noun phrase, as well as from
    conjunctions and commas to the items in a list.

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

    # Initialize a low-attention matrix with a bias for self-attention on some special tokens
    predicted_matrix = np.full((seq_len, seq_len), 0.05)
    np.fill_diagonal(predicted_matrix, 0.1)

    # Use spaCy to find noun phrase dependencies
    doc = nlp(sentence)
    
    # Map spaCy token indices to tokenizer indices
    def get_token_indices(spacy_idx):
        return [i for i, wid in enumerate(word_ids) if wid == spacy_idx]

    for token in doc:
        # Pattern 1: Adjectives and determiners binding to their head noun
        if token.pos_ in ["ADJ", "DET"] or token.dep_ == "prep":
            head_token = token.head
            # Check if the head is a noun or a part of a noun phrase
            if head_token.pos_ in ["NOUN", "PROPN", "VERB"]:
                modifier_indices = get_token_indices(token.i)
                head_indices = get_token_indices(head_token.i)
                for i in modifier_indices:
                    for j in head_indices:
                        predicted_matrix[i, j] = 0.5
        
        # Pattern 2: Conjunctions and punctuation binding to coordinated items
        if token.dep_ == "cc" or token.text in [",", " and", " or"]:
            if token.head:
                head_token = token.head
                # Check for conjuncts (items in a list)
                for child in head_token.children:
                    if child.dep_ == "conj":
                        conjunct_indices = get_token_indices(child.i)
                        modifier_indices = get_token_indices(token.i)
                        for i in modifier_indices:
                            for j in conjunct_indices:
                                predicted_matrix[i, j] = 0.4
                        
    # Normalize rows to sum to 1.0
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    predicted_matrix = predicted_matrix / row_sums

    return 'Adjectival and Noun Phrase Binding', predicted_matrix

# Example usage:
# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# sentence = "The gentle rustling of leaves, the chirping of crickets, and the distant hoot of an owl created a peaceful, calming ambiance."
# pattern_name, matrix = adjectival_and_noun_phrase_binding(sentence, tokenizer)
# print(pattern_name)
# print(matrix)