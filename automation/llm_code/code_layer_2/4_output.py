import numpy as np
import spacy

# Load the spaCy model.
# NOTE: You'll need to install it first: 'pip install spacy' and 'python -m spacy download en_core_web_sm'
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model. This will happen only once.")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def noun_modifier_pattern(sentence, tokenizer):
    """
    Hypothesizes the attention pattern for Layer 2, Head 4,
    based on Noun-Modifier Attention.

    This function identifies noun-modifier relationships in a sentence using spaCy's
    dependency parsing and generates a corresponding attention matrix.

    Args:
        sentence (str): The input sentence.
        tokenizer: A tokenizer compatible with BERT (e.g., from the Hugging Face library).

    Returns:
        tuple: A tuple containing the pattern name and the predicted attention matrix.
    """
    # Tokenize the sentence and get the sequence length
    tokenized_sentence = tokenizer.tokenize(sentence)
    token_ids = tokenizer.convert_tokens_to_ids(tokenized_sentence)
    
    # Add CLS and SEP tokens to the tokenized sentence and IDs
    tokenized_sentence = [tokenizer.cls_token] + tokenized_sentence + [tokenizer.sep_token]
    len_seq = len(tokenized_sentence)
    
    # Use spaCy to get the dependency parse tree
    doc = nlp(sentence)
    
    # Initialize the attention matrix with zeros
    predicted_matrix = np.zeros((len_seq, len_seq))

    # Get word-to-token mappings from the tokenizer
    # Hugging Face tokenizers provide a word_ids method for this
    token_word_ids = tokenizer(sentence, return_tensors="pt").word_ids()

    # Create a mapping from spaCy token index to BERT token index
    spacy_to_bert_map = {}
    current_spacy_index = 0
    
    for bert_token_idx, word_id in enumerate(token_word_ids):
        # Skip special tokens (None word_id)
        if word_id is not None:
            # If a new word has started, increment our spaCy index
            if current_spacy_index == 0 or word_id != token_word_ids[bert_token_idx - 1]:
                current_spacy_index = word_id
            spacy_to_bert_map[current_spacy_index] = bert_token_idx

    # Iterate through the spaCy dependency tree
    for token in doc:
        # Check if the token and its head are in the BERT token map
        if token.i + 1 in spacy_to_bert_map and token.head.i + 1 in spacy_to_bert_map:
            # Get the BERT token indices
            from_token_idx = spacy_to_bert_map[token.i + 1]
            to_token_idx = spacy_to_bert_map[token.head.i + 1]

            # We focus on modifiers attending to their heads.
            # Example: "old" (modifier) attends to "house" (head)
            # The attention is from the modifier to the head.
            # The 'dep' attribute describes the dependency relation (e.g., 'amod' for adjectival modifier).
            
            # The 'dep' attribute identifies the dependency relationship. We can use this to select relevant connections.
            # Common dependencies for modifiers: 'amod' (adjectival modifier), 'compound', 'det' (determiner)
            if token.dep_ in ['amod', 'compound', 'det', 'prep', 'pobj']:
                # Set a high attention weight for the from_token to the to_token
                predicted_matrix[from_token_idx, to_token_idx] = 1.0

            # Special case for conjunctions (e.g., 'and') and punctuation in lists
            # These tokens often attend to the head of the list.
            if token.dep_ in ['cc', 'punct']:
                if token.head.i + 1 in spacy_to_bert_map:
                    head_idx = spacy_to_bert_map[token.head.i + 1]
                    predicted_matrix[from_token_idx, head_idx] = 1.0
            
    # Normalize the matrix to create a valid attention pattern (sum of each row is 1)
    # Add a small epsilon to avoid division by zero
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-9 
    predicted_matrix = predicted_matrix / row_sums

    return "Noun-Modifier Attention Pattern", predicted_matrix