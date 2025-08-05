import numpy as np
import spacy
from typing import Tuple, List
from transformers import PreTrainedTokenizer
import nltk

nlp = spacy.load("en_core_web_sm")

def adjective_to_head_noun_attention(sentence: str, tokenizer: PreTrainedTokenizer) -> Tuple[str, np.ndarray]:
    """
    Hypothesizes the attention pattern for Layer 5, Head 4 as 'Adjective to Head Noun Attention'.

    This function creates a rule-encoded attention matrix where adjectives attend to the
    nouns they modify. It uses spaCy's dependency parser to identify these relationships.

    Parameters:
    - sentence (str): The input sentence.
    - tokenizer: A pre-trained BERT tokenizer instance.

    Returns:
    - tuple: A tuple containing the name of the pattern and the predicted
             attention matrix.
    """
    bert_tokens_and_ids = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        return_token_type_ids=False,
        return_attention_mask=False,
        return_tensors="pt"
    )
    token_ids = bert_tokens_and_ids['input_ids'][0]
    word_ids = bert_tokens_and_ids.word_ids(batch_index=0)
    len_seq = len(token_ids)
    
    predicted_matrix = np.zeros((len_seq, len_seq))

    # Use spaCy for linguistic analysis
    doc = nlp(sentence)
    
    # Iterate through the spaCy tokens to find adjectives and their heads
    for token in doc:
        # Check if the token is an adjective and has a head that is a noun
        if token.pos_ == 'ADJ' and token.head.pos_ in ['NOUN', 'PROPN', 'PRON']:
            adj_idx_spacy = token.i
            noun_idx_spacy = token.head.i

            # Map spaCy indices to BERT sub-token indices
            adj_bert_indices = [i for i, word_id in enumerate(word_ids) if word_id == adj_idx_spacy]
            noun_bert_indices = [i for i, word_id in enumerate(word_ids) if word_id == noun_idx_spacy]

            if adj_bert_indices and noun_bert_indices:
                # The first sub-token of the adjective attends to the first sub-token of the noun
                from_idx = adj_bert_indices[0]
                to_idx = noun_bert_indices[0]
                predicted_matrix[from_idx, to_idx] = 1.0

    # Add self-attention for [CLS] and [SEP]
    if len_seq > 1:
        predicted_matrix[0, 0] = 1.0
    if len_seq > 2:
        predicted_matrix[-1, -1] = 1.0
    
    # Normalize rows to ensure they sum to 1
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    normalized_matrix = predicted_matrix / row_sums

    return 'Adjective to Head Noun Attention Pattern', normalized_matrix