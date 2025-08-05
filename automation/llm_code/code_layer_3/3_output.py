import numpy as np
import spacy
from typing import Tuple

def punctuation_centric_pattern(sentence: str, tokenizer) -> Tuple[str, np.ndarray]:
    """
    Predicts the attention pattern for Layer 3, Head 2, which is responsible
    for the 'Punctuation-Centric Pattern'.

    This function identifies punctuation marks and creates an attention matrix
    where punctuation tokens have strong self-attention and also attend to each other.
    Other tokens in the sentence are predicted to attend to the nearest punctuation.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer object (e.g., from Hugging Face).

    Returns:
        Tuple[str, np.ndarray]: A tuple containing the pattern name and the
                                predicted attention matrix.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    
    tokens = tokenizer([sentence], return_tensors="pt")
    token_len = len(tokens.input_ids[0])
    
    predicted_matrix = np.zeros((token_len, token_len), dtype=float)

    # Use tokenizer's word_ids to map spaCy tokens to BERT tokens
    word_ids = tokens.word_ids(batch_index=0)
    
    punctuation_bert_indices = []
    for i, token in enumerate(doc):
        # We also need to handle sub-word tokenization for punctuation
        if token.is_punct:
            bert_indices = [j for j, word_id in enumerate(word_ids) if word_id == i]
            punctuation_bert_indices.extend(bert_indices)
    
    # Add CLS and EOS tokens to the list of punctuation-like tokens
    # as they often serve a similar boundary-marking role
    if token_len > 0:
        punctuation_bert_indices.append(0)  # CLS token
    if token_len > 1:
        punctuation_bert_indices.append(token_len - 1)  # EOS token
    
    punctuation_bert_indices = sorted(list(set(punctuation_bert_indices)))

    # Rule 1: Punctuation tokens attend heavily to themselves and to each other
    for from_idx in punctuation_bert_indices:
        for to_idx in punctuation_bert_indices:
            predicted_matrix[from_idx, to_idx] += 0.5 if from_idx == to_idx else 0.2
            
    # Rule 2: Other tokens attend to the nearest preceding and following punctuation
    for i in range(token_len):
        if i not in punctuation_bert_indices:
            # Find nearest preceding punctuation
            preceding_punct = [p for p in punctuation_bert_indices if p < i]
            if preceding_punct:
                nearest_preceding_idx = max(preceding_punct)
                predicted_matrix[i, nearest_preceding_idx] += 0.4
            
            # Find nearest following punctuation
            following_punct = [p for p in punctuation_bert_indices if p > i]
            if following_punct:
                nearest_following_idx = min(following_punct)
                predicted_matrix[i, nearest_following_idx] += 0.4

    # Normalize the matrix rows so they sum to 1
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero
    predicted_matrix = np.divide(predicted_matrix, row_sums, out=np.zeros_like(predicted_matrix), where=row_sums != 0)

    return 'Punctuation-Centric Pattern', predicted_matrix