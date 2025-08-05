import numpy as np
import spacy
from transformers import AutoTokenizer

def possessive_co_reference(sentence, tokenizer):
    """
    Predicts attention patterns for Layer 2, Head 10 based on possessive pronoun
    co-reference.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer to use (e.g., from a Hugging Face model).

    Returns:
        tuple: A tuple containing the pattern name and the predicted attention matrix.
    """
    
    try:
        # Load the English language model from spacy
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # If the model is not downloaded, download it and then load it
        print("Downloading spacy model 'en_core_web_sm'...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    # Tokenize the sentence with the provided tokenizer to get token IDs and offsets
    tokenized_sentence = tokenizer(sentence, return_tensors="pt")
    input_ids = tokenized_sentence.input_ids[0]
    
    # Use spacy to get linguistic information
    doc = nlp(sentence)
    
    # Initialize a blank attention matrix with size (token_len, token_len)
    token_len = len(input_ids)
    predicted_matrix = np.zeros((token_len, token_len))
    
    # Map spacy tokens to BERT tokens. This is crucial for aligning
    # the linguistic analysis with the attention matrix dimensions.
    word_ids = tokenized_sentence.word_ids(batch_index=0)
    
    # Create a list of all tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Find the positions of the [CLS] and [SEP] tokens
    cls_pos = 0
    sep_pos = token_len - 1

    # Apply a self-attention rule for CLS and SEP tokens
    # This is a common pattern in BERT and is included for a more valid matrix
    predicted_matrix[cls_pos, cls_pos] = 1.0
    predicted_matrix[sep_pos, sep_pos] = 1.0
    
    # Iterate through the spaCy tokens to find possessive pronouns and their antecedents
    for spacy_token in doc:
        # Check if the token is a possessive pronoun (e.g., "his", "her", "your")
        if spacy_token.pos_ == "PRON" and spacy_token.dep_ == "poss":
            
            # Find the position of the possessive pronoun in the BERT token list
            # We use the token's head (the noun it modifies) to find the co-referenced word.
            # spacy_token.head.i is the index of the noun the pronoun modifies.
            # spacy_token.i is the index of the possessive pronoun.

            # Get the word index in the original sentence for the pronoun and its head
            pronoun_word_idx = spacy_token.i
            antecedent_word_idx = spacy_token.head.i
            
            # Find all BERT token indices that correspond to the pronoun and its antecedent
            pronoun_bert_indices = [
                i for i, word_idx in enumerate(word_ids) if word_idx == pronoun_word_idx
            ]
            antecedent_bert_indices = [
                i for i, word_idx in enumerate(word_ids) if word_idx == antecedent_word_idx
            ]
            
            # Assign attention: each pronoun token attends to each antecedent token
            for p_idx in pronoun_bert_indices:
                for a_idx in antecedent_bert_indices:
                    # Setting attention weight to 1, indicating a strong link
                    predicted_matrix[p_idx, a_idx] = 1.0

            # Assign attention in the reverse direction to mirror bidirectional
            # co-reference patterns often seen in attention matrices.
            for a_idx in antecedent_bert_indices:
                for p_idx in pronoun_bert_indices:
                    predicted_matrix[a_idx, p_idx] = 1.0

    # For a more realistic matrix, we can add a small amount of self-attention
    # for all other tokens and normalize the matrix rows to sum to 1.
    for i in range(token_len):
        if np.sum(predicted_matrix[i, :]) == 0:
            predicted_matrix[i, i] = 1.0 # Self-attention for tokens with no links

    # Normalize each row to ensure the attention weights sum to 1.
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    predicted_matrix = predicted_matrix / np.where(row_sums == 0, 1, row_sums)
    
    return "Possessive Co-reference Pattern", predicted_matrix