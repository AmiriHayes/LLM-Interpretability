import numpy as np
import nltk
import spacy

# Ensure spaCy model is downloaded. This block will only execute if the model is not found.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Download necessary NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

def punctuation_list_quote_linking(sentence, tokenizer):
    """
    Predicts attention patterns for Layer 6, Head 2, focusing on
    punctuation-based linking within lists and quoted structures.

    Args:
        sentence (str): The input sentence.
        tokenizer: A HuggingFace tokenizer (e.g., AutoTokenizer.from_pretrained("bert-base-uncased")).
                   Must support `__call__` with `return_offsets_mapping=True` to get word_ids.

    Returns:
        tuple: A tuple containing the pattern name (str) and the predicted attention matrix (numpy.ndarray).
               The matrix size is (token_len * token_len), where token_len includes [CLS] and [SEP].
    """
    def get_subword_indices_for_word(word_idx, word_ids):
        """Helper to get all subword indices for a given original word index."""
        return [i for i, w_id in enumerate(word_ids) if w_id == word_idx]

    # Tokenize the sentence and get word_ids to map subword tokens back to original words.
    encoded_input = tokenizer(sentence, return_offsets_mapping=True, return_attention_mask=True, return_token_type_ids=True, return_tensors="pt")
    input_ids = encoded_input['input_ids'][0]
    word_ids = encoded_input.word_ids()
    token_len = len(input_ids)

    predicted_matrix = np.zeros((token_len, token_len))

    # Rule 1: Self-attention for [CLS] and [SEP] tokens
    predicted_matrix[0, 0] = 1.0 # [CLS] token
    predicted_matrix[token_len - 1, token_len - 1] = 1.0 # [SEP] token

    # Use NLTK for basic word tokenization to easily identify punctuation and conjunctions
    nltk_tokens = nltk.word_tokenize(sentence)

    # Stack to keep track of opening quotes for pairing
    quote_stack = [] # Stores subword indices of opening quotes

    # Keep track of the last colon encountered for list linking
    last_colon_subword_idx = -1

    # Iterate through each subword token
    for i in range(token_len):
        current_word_idx = word_ids[i]
        if current_word_idx is None: # Skip special tokens like CLS, SEP
            continue

        # Get the original word from NLTK tokens
        # Ensure current_word_idx is within bounds of nltk_tokens
        if current_word_idx < len(nltk_tokens):
            current_word = nltk_tokens[current_word_idx]
        else:
            continue # Should not happen if word_ids are correctly mapped

        # Rule 2: Handle quotation marks
        if current_word in ["'", '"', '`']: # Include backtick for robustness
            if not quote_stack: # It's an opening quote
                quote_stack.append(i)
            else: # It's a closing quote
                # Try to find the matching quote. If not found, just push/pop.
                found_match = False
                for k in range(len(quote_stack) - 1, -1, -1):
                    # Check if the actual token matches the type of quote
                    if (tokenizer.decode(input_ids[quote_stack[k]]) == current_word):
                        opening_quote_subword_idx = quote_stack.pop(k) # Pop the matched quote
                        predicted_matrix[i, opening_quote_subword_idx] = 1.0 # Link closing to opening
                        predicted_matrix[opening_quote_subword_idx, i] = 1.0 # Bidirectional link for robustness
                        found_match = True
                        break
                if not found_match: # If no matching opening quote found, treat as an opening quote
                    quote_stack.append(i)


        # Rule 3: Handle colons
        if current_word == ':':
            predicted_matrix[i, i] = 1.0 # Self-attention for colon
            # Link colon to the token immediately preceding it (often the noun introducing a list)
            if i > 0 and word_ids[i-1] is not None:
                prev_word_idx = word_ids[i-1]
                prev_subword_indices = get_subword_indices_for_word(prev_word_idx, word_ids)
                for prev_s_idx in prev_subword_indices:
                    predicted_matrix[i, prev_s_idx] = 1.0
            last_colon_subword_idx = i # Store for list item linking

        # Rule 4: Handle commas and coordinating conjunctions in lists
        if current_word == ',':
            # Link comma to the nearest preceding comma or colon
            found_link = False
            for j in range(i - 1, 0, -1): # Search backwards from the token before the current comma
                prev_subword_word_idx = word_ids[j]
                if prev_subword_word_idx is None: continue
                if prev_subword_word_idx < len(nltk_tokens):
                    prev_nltk_word = nltk_tokens[prev_subword_word_idx]
                    if prev_nltk_word == ',' or prev_nltk_word == ':':
                        predicted_matrix[i, j] = 1.0
                        found_link = True
                        break
            # If no specific preceding punctuation link found, link to CLS (general sentence start)
            if not found_link:
                predicted_matrix[i, 0] = 1.0

        elif current_word.lower() in ['and', 'or', 'but']: # Case-insensitive check for conjunctions
            # If preceded by a comma, link to that comma
            found_prev_comma = False
            for j in range(i - 1, 0, -1):
                prev_subword_word_idx = word_ids[j]
                if prev_subword_word_idx is None: continue
                if prev_subword_word_idx < len(nltk_tokens) and nltk_tokens[prev_subword_word_idx] == ',':
                    predicted_matrix[i, j] = 1.0
                    found_prev_comma = True
                    break
            # If not preceded by a comma, link to the token immediately before it (for simple conjunctions or lists without final comma)
            if not found_prev_comma and i > 0 and word_ids[i-1] is not None:
                predicted_matrix[i, i-1] = 1.0

        # Rule 5: Tokens after a colon (list items) attend to the colon
        # This is a heuristic: if a token is a content word (not punctuation/conjunction itself)
        # and appears after a colon, it might attend to the colon.
        if last_colon_subword_idx != -1 and i > last_colon_subword_idx:
            # Check if current token is a content word (not a delimiter itself)
            if current_word not in ['.', '?', '!', ';', ',', ':', '"', "'", 'and', 'or', 'but']:
                predicted_matrix[i, last_colon_subword_idx] = 1.0


    # Normalization: Ensure each row sums to 1.0.
    for i in range(token_len):
        row_sum = np.sum(predicted_matrix[i, :])
        if row_sum > 0:
            predicted_matrix[i, :] /= row_sum
        else:
            # If a token has no specific attention rule applied, it defaults to self-attention.
            predicted_matrix[i, i] = 1.0

    return 'Punctuation-Based List and Quote Structure Linking Pattern', predicted_matrix