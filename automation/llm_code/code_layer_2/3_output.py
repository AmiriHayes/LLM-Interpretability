import numpy as np
import spacy

def punctuation_to_content_alignment(sentence: str, tokenizer) -> tuple[str, np.ndarray]:
    """
    Hypothesizes that Layer 2, Head 3 is responsible for the 'Punctuation-to-Content Alignment' pattern,
    where a variety of tokens in a sentence attend to the final punctuation mark.
    This function predicts this attention pattern for a given sentence.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer (e.g., from the Hugging Face library) used for the model.

    Returns:
        tuple[str, np.ndarray]: A tuple containing the name of the pattern and the
                                predicted attention matrix of size (token_len x token_len).
    """
    # Load a smaller SpaCy model, as it's sufficient for this task.
    # The first time this runs, it might need to download the model.
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading 'en_core_web_sm' model for spaCy...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    tokens = tokenizer([sentence], return_tensors="pt")
    input_ids = tokens['input_ids'][0]
    token_len = len(input_ids)
    
    # Initialize a zero matrix for attention predictions.
    predicted_matrix = np.zeros((token_len, token_len))

    # Identify the index of the last token, which is often a punctuation mark.
    # We can use the tokenizer to get the token, or find the last token
    # that isn't padding or a special token.
    last_token_idx = token_len - 1
    
    # We will assume the last token is the punctuation mark we are interested in.
    # If the last token is not a punctuation mark, we can use spaCy to find the last punctuation
    # in the original sentence and map it to a token index, but for this hypothesis,
    # we'll assume the final token is the target.
    doc = nlp(sentence)
    
    punctuation_token_index = None
    last_word_token_index = None

    # Use spaCy to find the last punctuation token and the last word token.
    # This helps handle cases where the last token in the sentence is not
    # the last token in the tokenization sequence (e.g., a subword).
    for i, token in enumerate(doc):
        # We need to map spaCy's tokens back to the BERT tokenizer's tokens.
        # This is a simplification, but will work for most cases.
        if token.is_punct:
            punctuation_token_index = last_token_idx # Simplified mapping
        
        if not token.is_punct:
            last_word_token_index = i + 1 # Simplified mapping

    # Set attention from all tokens to the last token (the punctuation).
    # We'll normalize the attention so that each row sums to 1.
    for i in range(1, last_token_idx):
        predicted_matrix[i, last_token_idx] = 1.0

    # Handle the self-attention for the [CLS] token and the final punctuation.
    predicted_matrix[0, 0] = 1.0  # [CLS] token attends to itself.
    predicted_matrix[last_token_idx, last_token_idx] = 1.0 # Last token attends to itself.

    # Normalize the matrix to ensure each row's values sum to 1.
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero for rows that are all zeros.
    predicted_matrix = np.divide(predicted_matrix, row_sums, out=np.zeros_like(predicted_matrix), where=row_sums != 0)

    return 'Punctuation-to-Content Alignment', predicted_matrix