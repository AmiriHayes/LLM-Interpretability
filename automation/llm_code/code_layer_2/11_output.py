import numpy as np
from transformers import BertTokenizer
import spacy

# Load a small spaCy model for part-of-speech and dependency parsing.
# You may need to run `python -m spacy download en_core_web_sm` once.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def predict_L2H11(sentence: str, tokenizer: BertTokenizer) -> tuple[str, np.ndarray]:
    """
    Predicts the attention matrix for Layer 2, Head 11 based on the
    "Comma-to-Conjunction Linking" hypothesis.

    The function identifies commas that directly precede a coordinating
    conjunction (e.g., 'and', 'or') and assigns high attention from
    the comma token to the conjunction token.

    Args:
        sentence (str): The input sentence.
        tokenizer (BertTokenizer): The tokenizer for the model.

    Returns:
        tuple[str, np.ndarray]: A tuple containing the name of the pattern
                                and the predicted attention matrix.
    """
    # Tokenize the sentence and get word IDs to map to spaCy tokens
    tokens = tokenizer([sentence], return_tensors="pt")
    input_ids = tokens['input_ids'][0]
    word_ids = tokens.word_ids(batch_index=0)
    seq_len = len(input_ids)
    
    # Initialize a zero matrix for attention weights
    predicted_matrix = np.zeros((seq_len, seq_len), dtype=np.float32)

    # Use spaCy to parse the sentence for linguistic information
    doc = nlp(sentence)
    
    # Iterate through spaCy tokens to find the pattern
    # The word_ids array maps BERT tokens back to spaCy token indices.
    # We need to find the BERT token index for each spaCy token.
    spacy_token_to_bert_token_map = {}
    for i, word_id in enumerate(word_ids):
        if word_id is not None:
            if word_id not in spacy_token_to_bert_token_map:
                spacy_token_to_bert_token_map[word_id] = []
            spacy_token_to_bert_token_map[word_id].append(i)

    # Find the attention pattern based on linguistic rules
    attention_connections = []
    for token in doc:
        # Check for the pattern: a comma followed by a coordinating conjunction
        if token.text == ',' and token.i + 1 < len(doc) and doc[token.i + 1].pos_ == 'CCONJ':
            from_token_id = spacy_token_to_bert_token_map.get(token.i)
            to_token_id = spacy_token_to_bert_token_map.get(token.i + 1)
            
            if from_token_id and to_token_id:
                # Assign a high attention value from the comma to the first sub-token
                # of the conjunction. We use a value > 0 and normalize later.
                predicted_matrix[from_token_id[0], to_token_id[0]] = 1.0

    # Add self-attention for [CLS] token and attention to [CLS] for [SEP]
    # This is a common pattern for BERT's attention heads.
    predicted_matrix[0, 0] = 1.0
    predicted_matrix[seq_len - 1, 0] = 1.0
    predicted_matrix[seq_len - 1, seq_len - 1] = 1.0

    # Normalize each row to sum to 1 to represent a probability distribution.
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero for rows that have no attention
    row_sums[row_sums == 0] = 1.0
    normalized_matrix = predicted_matrix / row_sums

    return "Comma-to-Conjunction Linking", normalized_matrix