import numpy as np
import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def subject_anaphora_pattern(sentence, tokenizer):
    """
    Predicts the attention matrix for Layer 1, Head 9 based on the
    "Subject-Anaphora Pattern" hypothesis.

    This function identifies the main subject of a sentence and
    assigns high attention weights to that subject from other tokens,
    reflecting a pattern of anaphora and thematic linking.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer object (e.g., from Hugging Face).

    Returns:
        tuple: A tuple containing a string for the pattern name and a
               predicted attention matrix (size: token_len x token_len).
    """

    # Get tokens and their word IDs
    toks = tokenizer([sentence], return_tensors="pt")
    input_ids = toks.input_ids[0]
    word_ids = toks.word_ids(batch_index=0)
    token_len = len(input_ids)
    
    # Initialize an empty attention matrix
    predicted_matrix = np.zeros((token_len, token_len))

    # Get linguistic information from spaCy
    doc = nlp(sentence)
    
    # Find the main subject or anaphoric reference
    # Iterate through tokens to find the primary subject
    subject_token_index = -1
    for token in doc:
        # Check if the token is a subject (nsubj) or a pronoun
        if token.dep_ == "nsubj" or token.pos_ == "PRON":
            # Find the corresponding token index for the subject word
            for i in range(len(word_ids)):
                if word_ids[i] is not None and word_ids[i] == token.i:
                    subject_token_index = i
                    break
        if subject_token_index != -1:
            break

    # If a subject is found, assign attention weights to it
    if subject_token_index != -1:
        # Assign high attention from all other tokens to the subject
        for i in range(token_len):
            # Exclude self-attention for the subject token itself
            if i != subject_token_index:
                predicted_matrix[i, subject_token_index] = 1
    
    # Add self-attention for [CLS] and [SEP] tokens
    # and a placeholder for when a subject isn't found
    for i in range(token_len):
        if i == 0 or i == token_len - 1 or subject_token_index == -1:
            predicted_matrix[i, i] = 1

    # Normalize the matrix by row to make attention sum to 1
    # Check for rows with a sum of 0 to avoid division by zero
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Replace zero sums with 1 for normalization
    predicted_matrix = predicted_matrix / row_sums

    return "Subject-Anaphora Pattern", predicted_matrix

# Example usage (assuming a tokenizer is already defined)
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# sentence_example = "She wondered, 'Will he ever understand the complexities of this intricate problem?'"
# name, matrix = subject_anaphora_pattern(sentence_example, tokenizer)
# print(f"Pattern Name: {name}")
# print("Predicted Matrix:")
# print(matrix)