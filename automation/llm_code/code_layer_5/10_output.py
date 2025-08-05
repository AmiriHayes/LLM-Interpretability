import numpy as np
import spacy
from typing import Tuple
from transformers import PreTrainedTokenizer

# Load a simple spaCy model for tokenization and POS tagging
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def punctuation_backward_linking(sentence: str, tokenizer: PreTrainedTokenizer) -> Tuple[str, np.ndarray]:
    """
    Hypothesizes the attention pattern for Layer 5, Head 10 as a 'Backwards Punctuation-to-Content' pattern.

    This function creates a rule-encoded attention matrix where punctuation marks attend to
    the content token immediately preceding them. This reflects a local, backward-looking
    attention mechanism, likely to establish linguistic boundaries or context.

    Parameters:
    - sentence (str): The input sentence.
    - tokenizer: A pre-trained BERT tokenizer instance.

    Returns:
    - tuple: A tuple containing the name of the pattern and the predicted
             attention matrix.
    """
    # Tokenize sentence using the provided tokenizer to match the data's token indices
    token_ids = tokenizer.encode(sentence, add_special_tokens=True)
    len_seq = len(token_ids)
    predicted_matrix = np.zeros((len_seq, len_seq))

    # Use spaCy for linguistic information (is_punct)
    doc = nlp(sentence)
    spacy_tokens = [token.text for token in doc]
    spacy_is_punct = [token.is_punct for token in doc]

    # Align BERT tokens with spaCy tokens
    bert_word_ids = tokenizer.convert_ids_to_tokens(token_ids)
    
    # Iterate through BERT tokens
    # Note: Using `doc` for linguistic features, but `tokenizer` for token indices
    # This alignment can be tricky, but we can assume a simplified alignment for this task.
    # We will prioritize direct tokens and their positions.
    
    # A more robust alignment would use word_ids from the tokenizer,
    # but a simple mapping is sufficient for this hypothesis.
    
    # Simple heuristic to identify punctuation tokens
    punctuation_symbols = set(['.', ',', '?', '!', ':', ';', "'", '"'])
    
    # Loop through the BERT tokens
    for i in range(1, len_seq - 1): # Exclude CLS and SEP
        current_token = bert_word_ids[i].replace("##", "")
        # Heuristic: If current token is punctuation, it attends to the previous token
        if current_token in punctuation_symbols:
            predicted_matrix[i, i - 1] = 1.0

    # Ensure CLS and SEP tokens have some attention
    # [CLS] looks at the first word, and [SEP] looks at the last word
    if len_seq > 1:
        predicted_matrix[0, 1] = 1.0
    if len_seq > 2:
        predicted_matrix[-1, -2] = 1.0
    
    # Normalize the matrix rows so each row sums to 1.
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    normalized_matrix = predicted_matrix / row_sums

    return 'Backwards Punctuation-to-Content Pattern', normalized_matrix