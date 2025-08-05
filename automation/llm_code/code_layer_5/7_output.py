import numpy as np
from typing import List, Tuple
from transformers import PreTrainedTokenizer

def phrase_internal_linking(sentence: str, tokenizer: PreTrainedTokenizer) -> Tuple[str, np.ndarray]:
    """
    Hypothesizes the attention pattern for Layer 5, Head 7 as a 'Phrase-Internal Linking' pattern.

    This function predicts attention from a token to its immediate successor, with a higher
    likelihood if the first token is a determiner, preposition, or conjunction, and the
    second token is a content word (noun, verb, adjective).

    Parameters:
    - sentence (str): The input sentence.
    - tokenizer: A pre-trained BERT tokenizer instance.

    Returns:
    - tuple: A tuple containing the name of the pattern and the predicted
             attention matrix.
    """
    toks = tokenizer([sentence], return_tensors="pt", add_special_tokens=True)
    input_ids = toks.input_ids[0]
    word_ids = toks.word_ids(batch_index=0)
    len_seq = len(input_ids)
    out = np.zeros((len_seq, len_seq))

    # A simple list of parts of speech that tend to be function words
    # and often precede the word they link to in this pattern.
    function_words_to_link_next = {
        'the', 'a', 'an',  # Determiners
        'of', 'in', 'on', 'with', 'to', 'for', 'at', 'from', 'as', 'by', # Prepositions
        'and', 'but', 'or', # Conjunctions
    }

    # Get the tokens and their text representations from the tokenizer
    token_strings = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Iterate through tokens to find forward-looking links
    for i in range(1, len_seq - 1): # Exclude CLS and SEP
        current_token = token_strings[i]
        next_token = token_strings[i+1]

        # Prioritize attention from sub-word tokens to their next token
        if word_ids[i] is not None and word_ids[i] == word_ids[i+1]:
            out[i, i+1] = 1.0
        
        # Check for function words linking to the next token
        if current_token.lower() in function_words_to_link_next:
            out[i, i+1] = 1.0

        # Heuristic for linking an adjective to a noun
        if i + 2 < len_seq and current_token.isalpha() and token_strings[i+2].isalpha():
            if current_token.endswith('ing') or current_token.endswith('ed'):
                out[i, i+1] = 1.0

    # Ensure self-attention for special tokens
    out[0, 0] = 1.0  # CLS
    out[-1, -1] = 1.0 # SEP
    
    # Normalize matrix by row to represent a distribution
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    predicted_matrix = out / row_sums

    return 'Phrase-Internal Linking Pattern', predicted_matrix